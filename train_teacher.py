from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train
from helper.loops import validate_vanilla as validate
import warnings
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--CosineAnnealingLR', dest='CosineAnnealingLR', action='store_true',help='use CosineAnnealingLR in learning rate')
    parser.add_argument('--clip_grad', dest='clip_grad', action='store_true',help='use clip_grad when training backward')
    parser.add_argument('--smoothlabel', dest='smoothlabel', action='store_true',help='use smooth label for better performance')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'ResNet34','ResNet50', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16linerbn',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'vitbase'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','imagenet','cifar10'], help='dataset')
    parser.add_argument('--datapath', type=str, default='', help='path of dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

    ### distributed training
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')                        

    parser.add_argument('-o', '--output_dir', default='res', type=str, metavar='PATH',
                    help='path to save results')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    # if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    #     opt.learning_rate = 0.01

    opt.model_path = os.path.join(opt.output_dir,'models')
    opt.tb_path = os.path.join(opt.output_dir,'tensorboard')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    if len(iterations) == 1:
        lr_decay_step = int(iterations[0])
        opt.lr_decay_epochs = torch.arange(1,opt.epochs+1,lr_decay_step).numpy().tolist()
    else:
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    cfg_path = os.path.join(opt.save_folder,'argument.txt')
    with open(cfg_path, 'w') as f:
        for key, value in vars(opt).items():
            f.write('%s:%s\n'%(key, value))
            print(key, value)
    num_cls_dict = {'cifar10':10,'cifar100':100,'imagenet':1000}
    opt.n_cls = num_cls_dict[opt.dataset]

    return opt

def main():
    opt = parse_option()

    if opt.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    best_acc = 0
    opt.gpu = gpu

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))


    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
    # model
    model = model_dict[opt.model](num_classes=opt.n_cls)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opt.batch_size = int(opt.batch_size / ngpus_per_node)
            opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
   
    criterion = nn.CrossEntropyLoss().cuda(opt.gpu)
    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    if opt.CosineAnnealingLR:
        opt.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.epochs)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            if opt.gpu is None:
                checkpoint = torch.load(opt.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(opt.gpu)
                checkpoint = torch.load(opt.resume, map_location=loc)
            opt.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['accuracy'] if hasattr(checkpoint,'accuracy') else 0
            try:
                model.load_state_dict(checkpoint['model'])
            except:
                ## load distributed training model
                model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("warning: no optimizer find in resume model! ")
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            exit(1)

    cudnn.benchmark = True

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, train_sampler = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,distributed=opt.distributed)
    elif opt.dataset == 'cifar10':
        train_loader, val_loader, train_sampler = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,distributed=opt.distributed)
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, train_sampler = get_imagenet_dataloader(opt, datapath= opt.datapath, batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)


    ## only evaluate
    if opt.evaluate:
        validate(val_loader, model, criterion, opt)
        return

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(opt.start_epoch+1, opt.epochs + 1):
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        if not opt.CosineAnnealingLR:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'accuracy': test_acc,
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()

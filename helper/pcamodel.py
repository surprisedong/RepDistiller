from pickle import FALSE
import torch
import torch.nn as nn
from distiller_zoo import PCALoss
from .util import load_teacher
from dataset.imagenet import get_subimagenet_dataloader
from dataset.cifar100 import get_subcifar100_dataloader
from dataset.cifar10 import get_subcifar10_dataloader

def build_model_s(opt):
    print("model_s & model_t have same structure but less channels in PCA mode, building model_s......")
    channel_list = []
    criterion_kd = nn.ModuleList([])

    model_t = load_teacher(opt.path_t, opt.n_cls)
    model_t.eval()
    if opt.dataset == 'imagenet':
        statloader = get_subimagenet_dataloader(datapath= opt.datapath, batch_size=256)
    elif opt.dataset == 'cifar100':
        statloader = get_subcifar100_dataloader(batch_size=2048)
    elif opt.dataset == 'cifar10':
        statloader = get_subcifar10_dataloader(batch_size=2048)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(statloader):
            if opt.model_t.lower().startswith('vgg'):
                feat_t,_ = model_t(inputs, is_feat=True,preact=opt.preact,alllayer=opt.alllayer)
            elif opt.model_t.lower().startswith('resnet'):
                feat_t,_ = model_t(inputs, is_feat=True,preact=opt.preact,alllayer=opt.alllayer)
            elif opt.model_t.lower().startswith('mobilenet'):
                feat_t,_ = model_t(inputs, is_feat=True,preact=opt.preact,alllayer=opt.alllayer)
            else:
                assert False, f'{opt.model_t} is unsupported model right now'
    
    for i,feat in enumerate(feat_t):
        criterion = PCALoss(eigenVar=opt.eigenVar,crit_type=opt.crit_type,loss_type=opt.loss_type,\
            channels_truncate=opt.channel_list[i] if opt.channel_list else None )
        featProj = criterion.projection(feat)
        channeltruncate = featProj.shape[1]
        channel_list.append(channeltruncate)
        criterion_kd.append(criterion)
    print(f'channel truncate after PCA: {channel_list}')

    return channel_list, criterion_kd

from pickle import FALSE
import torch
import torch.nn as nn
from distiller_zoo import PCALoss
from .util import load_teacher
from dataset.imagenet import get_subimagenet_dataloader
from dataset.cifar100 import get_subcifar100_dataloader

def build_model_s(opt):
    print("model_s & model_t have same structure but less channels in PCA mode, building model_s......")
    channel_list = []
    criterion_kd = nn.ModuleList([])

    model_t = load_teacher(opt.path_t, opt.n_cls)
    model_t.eval()
    if opt.dataset == 'imagenet':
        statloader = get_subimagenet_dataloader(datapath= opt.datapath, batch_size=256)
    elif opt.dataset == 'cifar100':
        statloader = get_subcifar100_dataloader(batch_size=256)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(statloader):
            if opt.model_t.startswith('vgg'):
                feat_t,_ = model_t(inputs, is_feat=True,preact=opt.preact,alllayer=opt.alllayer)
            elif opt.model_t.startswith('ResNet'):
                feat_t,_ = model_t(inputs, is_feat=True,preact=opt.preact,alllayer=opt.alllayer)
            else:
                assert False, 'unsupported model right now'
    
    for feat in feat_t:
        criterion = PCALoss(eigenVar=opt.eigenVar,pca_s=opt.pca_s)
        featProj = criterion.projection(feat)
        channeltruncate = featProj.shape[1]
        channel_list.append(channeltruncate)
        criterion_kd.append(criterion)
    print(f'channel truncate after PCA: {channel_list}')

    return channel_list, criterion_kd

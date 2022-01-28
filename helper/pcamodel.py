from pickle import FALSE
import torch
import torch.nn as nn
from distiller_zoo import PCALoss
from .util import load_teacher
from dataset.imagenet import get_subimagenet_dataloader

def build_model_s(opt):
    print("model_s & model_t have same structure but less channels in PCA mode, building model_s......")
    criterion_kd = nn.ModuleList([])
    channel_list = []

    model_t = load_teacher(opt.path_t, opt.n_cls)
    model_t.eval()
    if opt.dataset == 'imagenet':
        statloader = get_subimagenet_dataloader(datapath= opt.datapath, batch_size=opt.batch_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(statloader):
            if opt.model_t.startswith('ResNet'):
                feat_t, _ = model_t(inputs, is_feat=True,preact=opt.preact)
                feat_t = feat_t[:-1]###delete last feature map
            elif opt.model_t.startswith('MobileNet'):
                feat_t, _ = model_t(inputs, is_feat=True,preact=opt.preact)
            else:
                assert False, 'unsupported model right now'
    
    for feat in feat_t:
        criterion = PCALoss(eigenVar=opt.eigenVar,attention=opt.attention)
        featProj = criterion.projection(feat)
        channeltruncate = featProj.shape[1]
        channel_list.append(channeltruncate)
        criterion_kd.append(criterion)
    print(f'channel truncate after PCA: {channel_list}')

    return criterion_kd, channel_list

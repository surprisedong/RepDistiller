from pickle import FALSE
import torch
import torch.nn as nn
from distiller_zoo import PCALoss
from .util import load_teacher

def build_model_s(opt):
    print("model_s & model_t have same structure but less channels in PCA mode, building model_s......")
    criterion_kd = nn.ModuleList([])
    channel_list = []

    model_t = load_teacher(opt.path_t, opt.n_cls)
    data = torch.rand(opt.batch_size, 3, 32, 32) if opt.dataset == 'cifar100' else torch.rand(opt.batch_size, 3, 224, 224)
    model_t.eval()
    if opt.model_t.startswith('ResNet'):
        feat_t, _ = model_t(data, is_feat=True,preact=False)
        feat_t = feat_t[:-1]
    elif opt.model_t.startswith('MobileNet'):
        feat_t, _ = model_t(data, is_feat=True,preact=True)
    else:
        assert False, 'unsupported model right now'
    
    for feat in feat_t:
        criterion = PCALoss(eigenVar=opt.eigenVar,truncate=opt.truncate)
        featProj = criterion.projection(feat)
        channeltruncate = featProj.shape[1]
        channel_list.append(channeltruncate)
        criterion_kd.append(criterion)
    print(f'channel truncate after PCA: {channel_list}')

    return criterion_kd, channel_list

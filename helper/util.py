from __future__ import print_function

import torch
import numpy as np
from models import model_dict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    try:
        try:
            model.load_state_dict(torch.load(model_path)['model'])
        except:
            model.load_state_dict(torch.load(model_path))##load official pretrained model
    except:
        ## load distributed training model
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path)['model'].items()})
    print('==> done')
    return model

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.unsqueeze(1), confidence) 
    return true_dist



class Focal_Loss(nn.Module):
    def __init__(self,weight=None,gamma=2):
        '''weight: 1D tensor, weight for loss of each class, used when sample is not balanced
           gamma : scale to learn difficult samples
        '''
        super(Focal_Loss,self).__init__()
        self.gamma = gamma
        self.weight = 1 if not weight else weight
    def forward(self,preds,labels):
        # ce = torch.nn.functional.cross_entropy(preds, labels, reduction='none')
        # y_pred = torch.exp(-ce)
        # floss=torch.pow((1-y_pred),self.gamma)*ce
        # floss=torch.mul(floss,self.weight)
        eps=1e-7
        preds = torch.softmax(preds,dim=-1)
        y_pred = preds.view((preds.size()[0],preds.size()[1])) #B X C

        try:
            target=labels.view(y_pred.size())
        except: 
            target = torch.zeros_like(y_pred)
            target.scatter_(1, labels.unsqueeze(1), 1)

        ce=-torch.log(y_pred+eps)*target
        floss=torch.pow((1-y_pred),self.gamma)*ce
        floss=torch.mul(floss,self.weight)
        floss=torch.sum(floss,dim=1)

        return torch.mean(floss)


if __name__ == '__main__':

    pass

"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math
from .util import Sequential_feat

__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ) if expand_ratio !=1 else \
            nn.Sequential(
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg
        stride_ = 1 if feature_dim in [10,100] else 2 ## cifar input size is 32,using stride 2 will perform bad

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, stride_],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, stride_)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(Sequential_feat(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, feature_dim),
        )


        self._initialize_weights()

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward_feat(self,x,preact=False):
        feat = []
        out = self.conv1(x)
        feat.append(out)

        out = self.blocks[0](out,is_feat=True)
        feat.extend(out)
        out = self.blocks[1](out[-1],is_feat=True)
        feat.extend(out)
        out = self.blocks[2](out[-1],is_feat=True)
        feat.extend(out)
        out = self.blocks[3](out[-1],is_feat=True)
        feat.extend(out)
        out = self.blocks[4](out[-1],is_feat=True)
        feat.extend(out)
        out = self.blocks[5](out[-1],is_feat=True)
        feat.extend(out)
        out = self.blocks[6](out[-1],is_feat=True)
        feat.extend(out)

        out = self.conv2(out[-1])
        feat.append(out)

        if not self.remove_avg:
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return feat, out


    def forward(self, x, is_feat=False, preact=False,alllayer=False):
        if alllayer:
            return self.forward_feat(x,preact=preact)

        out = self.conv1(x)
        f0 = out

        out = self.blocks[0](out)
        f1 = out
        out = self.blocks[1](out)
        f2 = out
        out = self.blocks[2](out)
        f3 = out
        out = self.blocks[3](out)
        f4 = out
        out = self.blocks[4](out)
        f5 = out
        out = self.blocks[5](out)
        f6 = out
        out = self.blocks[6](out)
        f7 = out

        out = self.conv2(out)
        f8 = out

        if not self.remove_avg:
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5, f6, f7, f8], out
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model

def mobilenetv2_raw(num_classes):
    return mobilenetv2_T_w(6, 1, num_classes)

def mobile_half(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)




class MobileNetV2PCA(MobileNetV2):
    def __init__(self, T,
                feature_dim,
                width_mult=1.,
                remove_avg=False,
                num_channels = []):
        super(MobileNetV2PCA, self).__init__(T,feature_dim,width_mult,remove_avg)
        stride_ = 1 if feature_dim in [10,100] else 2 ## cifar input size is 32,using stride 2 will perform bad
        num_channels_ = []
        num_channels_.append(num_channels[0])
        for n in [1,2,3,4,3,3,1]:
            num_channels_.append(num_channels[:n])
            del num_channels[:n]
        num_channels_.append(num_channels[-1])
        num_channels = num_channels_
        print(f'mobilenet num_channels:{num_channels}')
        
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, num_channels[1], 1, 1],
            [T, num_channels[2], 2, stride_],
            [T, num_channels[3], 3, 2],
            [T, num_channels[4], 4, 2],
            [T, num_channels[5], 3, 1],
            [T, num_channels[6], 3, 2],
            [T, num_channels[7], 1, 1],
    ]

        # building first layer
        input_channel = int(num_channels[0])
        self.conv1 = conv_bn(3, input_channel, stride_)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for i,(t, c, n, s) in enumerate(self.interverted_residual_setting):
            layers = []
            strides = [s] + [1] * (n - 1)
            for i,stride in enumerate(strides):
                output_channel = int(c[i] * width_mult)
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(Sequential_feat(*layers))

        self.last_channel = num_channels[8]
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.last_channel, feature_dim),
        )

        self._initialize_weights()


def mobilenetv2_T_w_pca(T, W, num_channels, feature_dim=100):
    model = MobileNetV2PCA(T=T, num_channels=num_channels ,feature_dim=feature_dim, width_mult=W)
    return model

def mobilenetv2_raw_pca(num_channels,num_classes):
    return mobilenetv2_T_w_pca(6, 1,num_channels, num_classes)

def mobile_half_pca(num_channels,num_classes):
    return mobilenetv2_T_w_pca(6, 0.5, num_channels, num_classes)







if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)

    net = mobile_half(100)

    feats, logit = net(x, is_feat=True, preact=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')


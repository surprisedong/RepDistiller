from .mobilenetv2 import mobilenetv2_raw, mobilenetv2_raw_pca
from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, \
    resnet110, resnet8x4, resnet32x4,resnet56PCA,resnet110PCA,resnet20PCA
from .resnetv2 import *
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn,vgg16liner_bn,vgg16liner_bnPCA,vgg16_bnPCA
from .mobilenetv2 import *
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .vit import vit_base

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet20PCA': resnet20PCA,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet56PCA':resnet56PCA,
    'resnet110PCA':resnet110PCA,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'ResNet34': ResNet34,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg16PCA':vgg16_bnPCA,
    'vgg19': vgg19_bn,
    'vgg16linerbn':vgg16liner_bn,
    'vgg16linerbnPCA':vgg16liner_bnPCA,
    'MobileNetV2': mobilenetv2_raw,
    'MobileNetV2PCA':mobilenetv2_raw_pca,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ResNet50PCA':ResNet50PCA,
    'ResNet34PCA':ResNet34PCA,
    'vitbase':vit_base
}

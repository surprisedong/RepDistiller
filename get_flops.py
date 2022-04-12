

from models import resnet56,resnet56PCA,mobilenet_v2,mobilenet_v2PCA,resnet20,resnet20PCA,resnet110,resnet110PCA,vgg16_bn,vgg16_bnPCA,resnet50,resnet50PCA
    
model_raw = resnet50(num_classes=1000)
model_pca = resnet50PCA(num_channels= [26, 64, 89, 116, 149, 218, 284, 325, 429, 552, 645, 704, 751, 788, 987, 916, 544], \
                         num_classes=1000)



from torchstat import stat
stat(model_raw, (3, 224, 224))
stat(model_pca, (3, 224, 224))
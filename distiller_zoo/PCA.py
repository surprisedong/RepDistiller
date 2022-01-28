import torch.nn as nn
import torch
from sklearn.decomposition import KernelPCA,PCA

class PCALoss(nn.Module):
    def __init__(self,eigenVar=1,attention=False,microBlockSz=1,channelsDiv=1,channels_truncate=None) -> None:
        super(PCALoss, self).__init__()
        self.eigenVar = eigenVar
        self.attention = attention
        self.microBlockSz = microBlockSz
        self.channelsDiv = channelsDiv
        self.collectStats = True
        self.channels_truncate = channels_truncate
        self.crit = nn.MSELoss()
    

    def forward(self,f_s,f_t):
        f_t = self.projection(f_t)
        assert f_t.shape == f_s.shape
        # Centering the f_s
        N, C, H, W = f_s.shape  # N x C x H x W
        f_s = PCALoss.featuresReshape(f_s, N, C, H, W, self.microBlockSz,self.channelsDiv) # C * (N * H * W)
        mn = torch.mean(f_s, dim=1, keepdim=True)
        f_s = f_s - mn
        f_s = PCALoss.featuresReshapeBack(f_s, N, C, H, W, self.microBlockSz,self.channelsDiv)

        if self.attention:
            return sum([(self.s[c] / torch.sum(self.s) * self.crit(f_s[:,c,:,:],f_t[:,c,:,:])) for c in range(len(self.s))])
        else:
            return self.crit(f_s,f_t)
            


    def projection(self, img, restore=False):
        N, C, H, W = img.shape  # N x C x H x W
        img = PCALoss.featuresReshape(img, N, C, H, W, self.microBlockSz,self.channelsDiv) # C * (N * H * W)
        mn = torch.mean(img, dim=1, keepdim=True)
        # Centering the data
        img = img - mn

        if self.collectStats:
            self.u, self.s = self.get_projection_matrix(img, self.eigenVar)
            self.u = self.u.detach()
            self.s = self.s.detach()
            #self.collectStats = False ##use same u&s will affect performance 

        imProj = torch.matmul(self.u.t(), img)
        if restore:
            imProj = torch.matmul(self.u, imProj) ##restore ori img

            # Bias Correction
            imProj = imProj - torch.mean(imProj, dim=1, keepdim=True)
            #self.mse = torch.sum((imProj - img) ** 2)
            # return original mean
            imProj = imProj + mn

            # return to general
            imProj = PCALoss.featuresReshapeBack(imProj, N, C, H, W, self.microBlockSz,self.channelsDiv)
        
        else:
            imProj = PCALoss.featuresReshapeBack(imProj, N, self.channels_truncate, H, W, self.microBlockSz,self.channelsDiv)

        return imProj

    def get_projection_matrix(self, im, eigenVar):
        # covariance matrix
        cov = torch.matmul(im, im.t()) / im.shape[1]
        # svd
        u, s, _ = torch.svd(cov)
        if eigenVar < 1:
            # find index where eigenvalues are more important
            if not self.channels_truncate:
                sRatio = torch.cumsum(s, 0) / torch.sum(s)
                cutIdx = (sRatio >= eigenVar).nonzero()[0]
                self.channels_truncate = cutIdx
            # throw unimportant eigenvector
            u = u[:, :self.channels_truncate]
            s = s[:self.channels_truncate]

        return u, s

    @staticmethod
    def featuresReshape(input, N, C, H, W, microBlockSz, channelsDiv):
        # check input
        if (microBlockSz > H):
            microBlockSz = H
        if (channelsDiv > C):
            channelsDiv = C
        assert (C % channelsDiv == 0)
        Ct = C // channelsDiv
        featureSize = microBlockSz * microBlockSz * Ct

        input = input.view(-1, Ct, H, W)  # N' x Ct x H x W
        input = input.permute(0, 2, 3, 1)  # N' x H x W x Ct
        input = input.contiguous().view(-1, microBlockSz, W, Ct).permute(0, 2, 1, 3)  # N'' x W x microBlockSz x Ct
        input = input.contiguous().view(-1, microBlockSz, microBlockSz, Ct).permute(0, 3, 2,1)  # N''' x Ct x microBlockSz x microBlockSz

        return input.contiguous().view(-1, featureSize).t()

    @staticmethod
    def featuresReshapeBack(input, N, C, H, W, microBlockSz, channelsDiv):
        if (microBlockSz > H):
            microBlockSz = H
        if (channelsDiv > C):
            channelsDiv = C
        assert (C % channelsDiv == 0)

        input = input.t()
        Ct = int(C / channelsDiv)

        input = input.reshape(-1, Ct, microBlockSz, microBlockSz).permute(0, 3, 2, 1)  # N'''  x microBlockSz x microBlockSz x Ct
        input = input.contiguous().view(-1, H, microBlockSz, Ct).permute(0, 2, 1, 3)  # N''  x microBlockSz x H x Ct
        input = input.contiguous().view(-1, H, W, Ct).permute(0, 3, 1, 2)  # N' x Ct x H x W X
        input = input.contiguous().view(N, C, H, W)  # N x C x H x W

        return input

    

class KPCALoss(PCALoss):
    def __init__(self, eigenVar=1, attention=False, microBlockSz=1, channelsDiv=1) -> None:
        super().__init__(eigenVar, attention, microBlockSz, channelsDiv)


    def projection(self, img):
        N, C, H, W = img.shape  # N x C x H x W
        img = KPCALoss.featuresReshape(img, N, C, H, W, self.microBlockSz,self.channelsDiv) # C * (N * H * W)
        img = img.t()
        if self.collectStats:
            scikit_kpca = KernelPCA(kernel='rbf', gamma=15)
            img_kpca = scikit_kpca.fit_transform(img)
            





        


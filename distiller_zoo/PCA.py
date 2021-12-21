import torch.nn as nn
import torch


class PCALoss(nn.modules):
    def __init__(self,eigenVar=1,truncate=False) -> None:
        super(PCALoss, self).__init__()
        self.eigenVar = eigenVar
        self.truncate = truncate
        self.microBlockSz = 1
        self.channelsDiv = 1
        self.collectStats = True

    def forward(self,f_s,f_t):
        f_t = self.projection(f_t)
        assert f_t.shape == f_s.shape
        return nn.MSELoss(f_s,f_t)


    def projection(self, img):
        N, C, H, W = img.shape  # N x C x H x W
        img = featuresReshape(img, N, C, H, W, self.microBlockSz,self.channelsDiv) # c * (n * h * w)
        self.channels = img.shape[0]

        mn = torch.mean(img, dim=1, keepdim=True)
        # Centering the data
        img = img - mn

        if self.collectStats:
            self.u, self.s = get_projection_matrix(img, self.eigenVar, self.truncate)
            self.original_channels = self.u.shape[0]
            self.collectStats = False

        self.channels = self.u.shape[1]
        imProj = torch.matmul(self.u.t(), img)

        # return to general
        imProj = featuresReshapeBack(imProj, N, C, H, W, self.microBlockSz,self.channelsDiv)

        return imProj


def get_projection_matrix(im, eigenVar, truncate=False):
    # covariance matrix
    cov = torch.matmul(im, im.t()) / im.shape[1]
    # svd
    u, s, _ = torch.svd(cov)
    if truncate:
        # find index where eigenvalues are more important
        sRatio = torch.cumsum(s, 0) / torch.sum(s)
        cutIdx = (sRatio >= eigenVar).nonzero()[0]
        # throw unimportant eigenvector
        u = u[:, :cutIdx]
        s = s[:cutIdx]

    return u, s


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


def featuresReshapeBack(input, N, C, H, W, microBlockSz, channelsDiv):
    if (microBlockSz > H):
        microBlockSz = H
    if (channelsDiv > C):
        channelsDiv = C
    assert (C % channelsDiv == 0)

    input = input.t()
    Ct = C // channelsDiv

    input = input.view(-1, Ct, microBlockSz, microBlockSz).permute(0, 3, 2,1)  # N'''  x microBlockSz x microBlockSz x Ct
    input = input.contiguous().view(-1, H, microBlockSz, Ct).permute(0, 2, 1,3)  # N''  x microBlockSz x H x Ct
    input = input.contiguous().view(-1, H, W, Ct).permute(0, 3, 1, 2)  # N' x Ct x H x W X
    input = input.contiguous().view(N, C, H, W)  # N x C x H x W

    return input

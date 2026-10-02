import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np

import os

# from torchmetrics.metric import Metric

def gram_matrix(data):
    a, b, c, d = data.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = data.view(a, b, c * d)  # resise F_XL into \hat F_XL
    features_t = features.transpose(1,2)

    G = torch.matmul(features, features_t)  # compute the gram product

    del features
    del features_t

    return G


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, condition, target):
        
        G1 = gram_matrix(condition)
        G2 = gram_matrix(target)
        loss = self.criterion(G1,G2)# G2 need detach 
        
        return loss
    


# class TotalVariation(Metric):
#     """
#     Computes Total Variation loss.
#     Adapted from: https://github.com/jxgu1016/Total_Variation_Loss.pytorch
#     Args:
#         dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
#             before returning the value at the step.
#         compute_on_step: Forward only calls ``update()`` and returns None if this is set to
#             False.
#     """

#     is_differentiable = True
#     higher_is_better = False
#     current: torch.Tensor
#     total: torch.Tensor

#     def __init__(self, dist_sync_on_step: bool = False, compute_on_step: bool = True):
#         super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
#         self.add_state("current", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

#     def update(self, img: torch.Tensor) -> None:
#         """Update method for TV Loss.
#         Args:
#             img (torch.Tensor): A NCHW image batch.
#         Returns:
#             A loss scalar value.
#         """
#         _height = img.size()[2]
#         _width = img.size()[3]
#         _count_height = self.tensor_size(img[:, :, 1:, :]).cuda()
#         _count_width = self.tensor_size(img[:, :, :, 1:]).cuda()
#         _height_tv = torch.pow((img[:, :, 1:, :] - img[:, :, : _height - 1, :]), 2).sum().cuda()
#         _width_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, : _width - 1]), 2).sum().cuda()
#         self.current += 2 * (_height_tv / _count_height + _width_tv / _count_width)
#         self.total += img.numel()

#     def compute(self):
#         return self.current.float() / self.total

#     @staticmethod
#     def tensor_size(t):
#         return t.size()[1] * t.size()[2] * t.size()[3]


# class Vgg19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg = models.vgg19(pretrained=False)
#         vgg.load_state_dict(torch.load("./vgg19-dcbb9e9d.pth"))
#         vgg_pretrained_features = vgg.features
#         self.vgg = vgg
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out

#     def extract(self, x):
#         x = self.vgg.features(x)
#         x = self.vgg.avgpool(x)
#         return x

# class StyleLoss_2(nn.Module):
#     def __init__(self):
#         super(StyleLoss, self).__init__()
#         self.vgg = Vgg19().cuda()
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

#     def forward(self, x, y):
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
#         loss = 0
#         for i in range(len(x_vgg)):
#             N, C, H, W = x_vgg[i].shape
#             for n in range(N):
#                 phi_x = x_vgg[i][n]
#                 phi_y = y_vgg[i][n]
#                 phi_x = phi_x.reshape(C, H * W)
#                 phi_y = phi_y.reshape(C, H * W)
#                 G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
#                 G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
#                 loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)) * self.weights[i]
#         return loss

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) +
                        torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self, opt):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.opt = opt

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]
        Gxcenter = Gx[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gxup = Gx[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gxdown = Gx[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gxleft = Gx[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gxright = Gx[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        Gycenter = Gy[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gyup = Gy[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gydown = Gy[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gyleft = Gy[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gyright = Gy[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))

class WeightedMSE(nn.Module):
	def __init__(self, b_mult = 1.0):
		super(WeightedMSE, self).__init__()
		self.b_mult = b_mult

	def forward(self, y_true, y_pred):
		epsilon = 1e-07

		temp_y = y_true.clone()
		_ones = torch.ones(temp_y.shape).cuda()
		_zeros = torch.zeros(temp_y.shape).cuda()
		temp_y = torch.where(temp_y > 0.5, _ones, temp_y)
		temp_y = torch.where(temp_y <= 0.5, _zeros, temp_y)
		_mean = torch.mean(temp_y, dim=(1, 2, 3))

#		b = (torch.mean(y_true, dim=(1, 2, 3)) * self.b_mult).detach()

		_epsilon = (torch.ones(y_pred.shape) * epsilon).cuda()
		epsilon_ = (torch.ones(y_pred.shape) * (1-epsilon)).cuda()
		torch.where(y_pred < epsilon, _epsilon, y_pred)
		torch.where(y_pred > (1-epsilon), epsilon_, y_pred)

		zeros = torch.zeros(y_true.shape).cuda()
		ones = torch.ones(y_true.shape).cuda()
		
		# background covering
		zero_idx = torch.where(y_true==0, ones, zeros)
		zero_idx.type_as(torch.cuda.FloatTensor())
		m_neg = torch.mean(((y_pred ** 2) * zero_idx), dim=(1, 2, 3))

		# edges covering
		non_zero_idx = torch.where(y_true!=0, ones, zeros)
		non_zero_idx.type_as(torch.cuda.FloatTensor())
		m_pos = torch.mean(((y_pred - y_true) ** 2) * non_zero_idx, dim=(1, 2, 3))

		m = (1.0 - _mean) * m_pos + _mean * m_neg
		return torch.mean(m, dim=-1)

class cannyLoss(nn.Module):
	def __init__(self):
		super(cannyLoss, self).__init__()
		self.vgg = Vgg19()
		self.vgg.cuda()
		self.criterion = nn.L1Loss()
		self.lamdas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
		self.gammas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
		self.mse = WeightedMSE()
	def forward(self, x, y):
		self.vgg.zero_loss()
		x3 = torch.cat([x, x, x], 1)
		y3 = torch.cat([y, y, y], 1)
		self.vgg(x3, y3)
		style = self.vgg.get_style()
		self.vgg.zero_loss()
		mse = self.mse(x, y)
		return style, mse
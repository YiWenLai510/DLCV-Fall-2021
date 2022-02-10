import os
import random
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import entropy
from fid_score import calculate_fid_given_paths
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from PIL import Image


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True)#nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)#nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )
        # self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y
class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                (nn.Conv2d(in_dim, out_dim, 5, 2, 2)),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
            
        """ Medium: Remove the last sigmoid layer for WGAN. """
        self.ls = nn.Sequential(
            (nn.Conv2d(in_dim, dim, 5, 2, 2)), 
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            (nn.Conv2d(dim * 8, 1, 4)),
            nn.Sigmoid(), 
        )
        # self.apply(weights_init)
        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
np.random.seed(422)
torch.manual_seed(422)
torch.cuda.manual_seed(422)
random.seed(422)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,)
parser.add_argument('--z_dim', type=int, default=100,)
parser.add_argument('--model_path',default='')
parser.add_argument('--result_dir',default='p1_result')

args = parser.parse_args()
print(args)


# D = Discriminator(3)
G = Generator(args.z_dim)
# print(G)
# print(D)
G.load_state_dict(torch.load(args.model_path))
G.eval()
G.cuda()
# Generate 1000 images and make a grid to save them.
n_output = 1000
z_sample = Variable(torch.randn(n_output, args.z_dim)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0
# print(imgs_sample.shape)
# torchvision.utils.save_image(imgs_sample[:32], 'p1_result.jpg', nrow=8)
for i in range(1000):
    torchvision.utils.save_image(imgs_sample[i], os.path.join(args.result_dir,f'{i+1:04}.png'))
# fid = calculate_fid_given_paths(('p1_result','hw2_data/face/test'),batch_size=32 ,device=device, dims=2048, num_workers=1)
# print(fid)
    

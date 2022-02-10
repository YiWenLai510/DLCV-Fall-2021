import os
import random
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import glob
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from qqdm import qqdm
from scipy.stats import entropy
from torchvision.models.inception import inception_v3
from fid_score import calculate_fid_given_paths

writer = SummaryWriter()


class CrypkoDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.fnames = os.listdir(folder)
        self.num_samples = len(self.fnames)
        self.folder = folder
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(os.path.join(self.folder,fname))
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

# from spectral_normalization import SpectralNorm
import torch.nn.utils.spectral_norm as SpectralNorm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        self.apply(weights_init)

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

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

np.random.seed(422)
torch.manual_seed(422)
torch.cuda.manual_seed(422)
random.seed(422)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,)
parser.add_argument('--z_dim', type=int, default=100,)
parser.add_argument('--n_epochs', type=int, default=500,)
parser.add_argument('--n_critic', type=int, default=1,)
parser.add_argument('--lr', type=float, default=2e-4,)

parser.add_argument('--save_dir',default='p1_save_models')
parser.add_argument('--log_dir',default='p1_log')
parser.add_argument('--result_dir',default='p1_result')

args = parser.parse_args()
print(args)

ModelName = '_'.join(('p1_DCGAN_ModLabel', str(args.batch_size), str(args.n_epochs), str(args.n_critic),))   

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.result_dir, exist_ok=True)


compose = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

dataset = CrypkoDataset('hw2_data/face/train',transform=compose)
# DataLoader
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

# Training hyperparameters
z_sample = Variable(torch.randn(100, args.z_dim)).cuda()

# Model
G = Generator(in_dim=args.z_dim).cuda()
D = Discriminator(3).cuda()
# G.load_state_dict(torch.load('p1_DCGAN_ModLabel_128_250_1_246_G.pth'))
G.train()
D.train()

# Loss
criterion = nn.BCELoss()

# Optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

"""### Training loop
We store some pictures regularly to monitor the current performance of the Generator, and regularly record checkpoints.
"""
fid_dict = {}
incep_dict = {}
steps = 0

for e, epoch in enumerate(range(args.n_epochs)):
       
    for i, data in enumerate(dataloader):

        imgs = data
        imgs = imgs.cuda()
        bs = imgs.size(0)
        #  Train D 
        z = Variable(torch.randn(bs, args.z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)
        # Label
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()

        # 0.05 to flip label
        for r in r_label:
            if random.random() < 0.05:
                r = 0
        for f in f_label:
            if random.random() < 0.05:
                f = 1
        # soft label
        for r in r_label:
            if r == 1:
                r -= random.randint(0,3) * 0.1
            else:
                r += random.randint(0,3) * 0.1
        for f in f_label:
            if f == 0:
                f += random.randint(0,3) * 0.1
            else:
                f -= random.randint(0,3) * 0.1

        # Model forwarding
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        # Compute the loss for the discriminator.
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = (r_loss + f_loss) / 2
        # Model backwarding
        D.zero_grad()
        loss_D.backward()
        # Update the discriminator.
        opt_D.step()

        writer.add_scalar("Loss/Discriminator", loss_D, epoch)

        # ============================================
        #  Train G
        # ============================================
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()
        
        if steps % args.n_critic == 0:
            # Generate some fake images.
            z = Variable(torch.randn(bs, args.z_dim)).cuda()
            f_imgs = G(z)
            # Model forwarding
            f_logit = D(f_imgs)
            # Compute the loss for the generator.
            loss_G = criterion(f_logit, r_label)
            # Model backwarding
            G.zero_grad()
            loss_G.backward()
            # Update the generator.
            opt_G.step()
            writer.add_scalar("Loss/Generator", loss_G, epoch)

        steps += 1

    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(args.log_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)

    # Generate 1000 images and make a grid to save them.
    for iter in range(4):
        n_output = 250
        z_500 = Variable(torch.randn(n_output, args.z_dim)).cuda()
        imgs_sample = (G(z_500).data + 1) / 2.0
        
        for i in range(n_output):
            idx = i + n_output * iter + 1
            torchvision.utils.save_image(imgs_sample[i], os.path.join(args.result_dir,f'{idx}.jpg'))
    
    test_dataset = CrypkoDataset('p1_result',transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]))
    fid = calculate_fid_given_paths(('p1_result','hw2_data/face/test'),batch_size=32 ,device=device, dims=2048, num_workers=1)
    is_score = inception_score(test_dataset, cuda=True, batch_size=16, resize=True, splits=10)

    print(f'epoch:{epoch} fid:{fid} inception:{is_score[0]}')
    fid_dict[epoch] = fid
    incep_dict[epoch] = is_score
    # if (e+1) % 5 == 0 or e == 0:
        # Save the checkpoints.
    torch.save(G.state_dict(), os.path.join(args.save_dir, ModelName +'_'+str(epoch)+'_G.pth'))
        # torch.save(D.state_dict(), os.path.join(args.save_dir, ModelName +'_'+str(epoch)+'_D.pth'))
    G.train()

writer.flush()

print('best fid epoch & score',min(fid_dict,key=fid_dict.get),min(fid_dict.values()))
print('best is epoch & score',max(incep_dict,key=incep_dict.get),max(incep_dict.values()))
# ====================================================================================================
# """## Inference
# Use the trained model to generate anime faces!

# G = Generator(args.z_dim)
# G.load_state_dict(torch.load('p1_save_models/p1_DCGAN_ModLabel_128_250_1_246_G.pth'))
# G.eval()
# G.cuda()

# # Generate 1000 images and make a grid to save them.
# n_output = 1000
# z_sample = Variable(torch.randn(n_output, args.z_dim)).cuda()
# imgs_sample = (G(z_sample).data + 1) / 2.0
# # torchvision.utils.save_image(imgs_sample[:32], 'p1_result.jpg', nrow=8)

# for i in range(1000):
#     torchvision.utils.save_image(imgs_sample[i], os.path.join(args.result_dir,f'{i+1}.jpg'))
# # ====================================================================================================

# test_dataset = test_dataset = CrypkoDataset('p1_result',transform=transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ]))
# fid = calculate_fid_given_paths(('p1_result','hw2_data/face/test'),batch_size=32 ,device=device, dims=2048, num_workers=1)
# inception = inception_score(test_dataset, cuda=True, batch_size=32, resize=True, splits=10)
# print(fid,inception)
# ### Load model
# """
# for f in sorted(os.listdir('p1_save_models')):
#     if f[:-4].split('_')[6] == 'G':

#         epoch = f[:-4].split('_')[5]
#         G = Generator(args.z_dim)
#         G.load_state_dict(torch.load(os.path.join(args.save_dir, f)))
#         G.eval()
#         G.cuda()

#         # Generate 1000 images and make a grid to save them.
#         n_output = 1000
#         z_sample = Variable(torch.randn(n_output, args.z_dim)).cuda()
#         imgs_sample = (G(z_sample).data + 1) / 2.0
#         # torchvision.utils.save_image(imgs_sample[:32], 'p1_result.jpg', nrow=8)

#         for i in range(1000):
#             torchvision.utils.save_image(imgs_sample[i], os.path.join(args.result_dir,f'{i+1}.jpg'))
#         # ====================================================================================================

#         test_dataset = get_dataset('p1_result')
#         fid = calculate_fid_given_paths(('p1_result','hw2_data/face/test'),batch_size=32 ,device=device, dims=2048, num_workers=1)
#         inception = inception_score(test_dataset, cuda=True, batch_size=32, resize=True, splits=10)
#         print (f'epoch:{epoch}, fid:{fid}, inception:{inception[0]}')

# '''99 33.007362867991986 (2.1523181616557796, 0.11968742094307992)'''
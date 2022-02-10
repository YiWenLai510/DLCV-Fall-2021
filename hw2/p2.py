import argparse
import os
import random
import numpy as np
import math
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
from torchvision.utils import save_image
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
from qqdm import qqdm
# os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--save_dir',default='p2_save_models')
parser.add_argument('--log_dir',default='p2_log')
parser.add_argument('--result_dir',default='p2_result')

opt = parser.parse_args()
print(opt)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128, 0.8),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # gen_input = torch.cat((self.label_emb(labels), noise), dim=1)
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 2 #opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


class DigitDataset(nn.Module):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path,sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')
        self.images_folder = images_folder
        self.transform = transform
        # self.class2index = {"cat":0, "dog":1}
        # print(self.df.iloc[0]['image_name'])        
        # print(self.df.iloc[0]['label'])        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.iloc[index]['image_name']
        label = self.df.iloc[index]['label']
        image = Image.open(os.path.join(self.images_folder, filename))#.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class TestDataset(nn.Module):
    def __init__(self,  images_folder, transform = None):
        # self.df = pd.read_csv(csv_path,sep=r'\s*,\s*',
                        #    header=0, encoding='ascii', engine='python')
        self.images_folder = images_folder
        self.transform = transform
        self.img_list = os.listdir(self.images_folder)
        # self.class2index = {"cat":0, "dog":1}
        # print(self.df.iloc[0]['image_name'])        
        # print(self.df.iloc[0]['label'])        
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        filename = self.img_list[index]
        label = filename[:-4].split('_')[0]
        image = Image.open(os.path.join(self.images_folder, filename))#.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ModelName = '_'.join(('p2_ACGAN', str(opt.batch_size), str(opt.n_epochs)))   

os.makedirs(opt.save_dir, exist_ok=True)
os.makedirs(opt.log_dir, exist_ok=True)
os.makedirs(opt.result_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataset = DigitDataset( csv_path='hw2_data/digits/mnistm/train.csv', 
                        images_folder='hw2_data/digits/mnistm/train',
                        transform=transforms.Compose([transforms.Resize(opt.img_size),transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join(opt.log_dir,f"{epoch}_{batches_done}.png" ) , nrow=n_row, normalize=True)

def calculate_acc(generator):
    # evaluate
    classifier = Classifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier.load_state_dict(torch.load("Classifier.pth")['state_dict'])
    classifier = classifier.to(device)
    
    generator.eval()

    z = Variable(FloatTensor(np.random.normal(0, 1, (1000, opt.latent_dim))))
    labels = np.array([label for label in range(10) for j in range(100)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

    cnt = 0

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(size=28),
    #     transforms.ToTensor()
    # ])

    for label in range(10):
        for j in range(100):
            
            idx = label * 100 + j
            img = gen_imgs.data[idx]
            save_image(img, os.path.join(opt.result_dir,f"{label}_{j:03}.png" ),normalize=True) # need to resize to 5 * 5
            
    test_set = TestDataset('p2_result',transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
    for imgs, labels in testloader:

        labels = labels.to(device)
        logits = classifier(imgs.to(device))
        pred = torch.argmax(logits,dim=1)
        # print(len(pred == labels))
        cnt += (pred == labels).sum()

    generator.train()    

    return   (cnt/1000)


best_acc = 0
# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):

    # progress_bar = qqdm(dataloader)

    generator.train()
    discriminator.train()

    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)

        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
        g_loss.backward()
        optimizer_G.step()
       
        # ---------------------
        #  Train Discriminator
        # ---------------------

        for r in valid:
            if random.random() < 0.05:
                r = 0
        for f in fake:
            if random.random() < 0.05:
                f = 1
        # soft label
        for r in valid:
            if r == 1:
                r -= random.randint(0,3) * 0.1
            else:
                r += random.randint(0,3) * 0.1
        for f in fake:
            if f == 0:
                f += random.randint(0,3) * 0.1
            else:
                f -= random.randint(0,3) * 0.1

        optimizer_D.zero_grad()
        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        # progress_bar.set_infos({
        #     'Epoch': f'{epoch+1}/{opt.n_epochs}',
        #     'Batch': f'{i}/{len(dataloader)}',
        #     'Loss_D': round(d_loss.item(), 4),
        #     'Loss_G': round( g_loss.item(), 4),
        #     'Acc': 100 * d_acc
        # })

       
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done, epoch=epoch)

    acc = calculate_acc(generator)
    print(f'epoch:{epoch}, Classifier accuracy:{acc}')

    
    # Save the checkpoints.
    torch.save(generator.state_dict(), os.path.join(opt.save_dir, f'{ModelName}_{epoch}_G.pth'))
        # torch.save(discriminator.state_dict(), os.path.join(opt.save_dir, f'{ModelName}_{epoch}_D.pth'))
# accs = {}
# for f in sorted(os.listdir('p2_save_models')):
#     generator.load_state_dict(torch.load(os.path.join('p2_save_models',f)))
#     acc = calculate_acc(generator).cpu().item()
#     epoch = f[:-4].split('_')[4]
#     print(acc, epoch)    
#     accs[epoch] = acc
# max_path = f'p2_save_models/p2_ACGAN_128_100_{max(accs,key=accs.get)}_G.pth'
# print(max_path)
# generator.load_state_dict(torch.load('p2_save_models/p2_ACGAN_modLabel_128_100_89_G.pth'))
# print(calculate_acc(generator))
# generator.eval()

# sample_image(1,1,1)
# print('save')

# z = Variable(FloatTensor(np.random.normal(0, 1, (1000, opt.latent_dim))))
# labels = np.array([label for label in range(10) for j in range(100)])
# labels = Variable(LongTensor(labels))
# print(labels)
# gen_imgs = generator(z, labels)

# err_cnt = 0
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(size=28),
#     transforms.ToTensor()
# ])
# for label in range(10):
#     for j in range(100):

#         idx = label*100+j
#         img = transform(gen_imgs.data[idx])
#         save_image(img, os.path.join(opt.result_dir,f"{label}_{j:03}.png" ),normalize=True) # need to resize to 5 * 5
#         log = net(torch.unsqueeze(img,0).to(device))
#         pred = torch.argmax(log,1)
#         if pred != label:
#             err_cnt += 1
#         # if label ==0 and j ==0:
#         #     print(log)
#         #     print(pred)
#         #     print(label)
# print('classifier acc: ',1 - err_cnt/1000) # epoch err = 0.46

'''epoch:0, Classifier accuracy:0.26900002360343933                                                                                                                             │|                               |                      |                  N/A |
epoch:1, Classifier accuracy:0.2900000214576721                                                                                                                              │+-------------------------------+----------------------+----------------------+
epoch:2, Classifier accuracy:0.35200002789497375                                                                                                                             │
epoch:3, Classifier accuracy:0.44600000977516174                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:4, Classifier accuracy:0.49000000953674316                                                                                                                             │| Processes:                                                                  |
epoch:5, Classifier accuracy:0.5610000491142273                                                                                                                              │|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
epoch:6, Classifier accuracy:0.5890000462532043                                                                                                                              │|        ID   ID                                                   Usage      |
epoch:7, Classifier accuracy:0.6290000081062317                                                                                                                              │|=============================================================================|
epoch:8, Classifier accuracy:0.6570000052452087                                                                                                                              │|    0   N/A  N/A      1358      G   /usr/lib/xorg/Xorg                134MiB |
epoch:9, Classifier accuracy:0.6880000233650208                                                                                                                              │|    0   N/A  N/A      2303      G   /usr/lib/xorg/Xorg                239MiB |
epoch:10, Classifier accuracy:0.6890000104904175                                                                                                                             │|    0   N/A  N/A      2471      G   /usr/bin/gnome-shell               10MiB |
epoch:11, Classifier accuracy:0.7100000381469727                                                                                                                             │|    0   N/A  N/A      2897      G   ...AAAAAAAAA= --shared-files        4MiB |
epoch:12, Classifier accuracy:0.7120000123977661                                                                                                                             │|    0   N/A  N/A   1347899      C   python3                          3359MiB |
epoch:13, Classifier accuracy:0.6790000200271606                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:14, Classifier accuracy:0.7390000224113464                                                                                                                             │
epoch:15, Classifier accuracy:0.7330000400543213                                                                                                                             │
epoch:16, Classifier accuracy:0.7560000419616699                                                                                                                             │
epoch:17, Classifier accuracy:0.7500000596046448                                                                                                                             │
epoch:18, Classifier accuracy:0.6970000267028809                                                                                                                             │
epoch:19, Classifier accuracy:0.7350000143051147                                                                                                                             │
epoch:20, Classifier accuracy:0.7700000405311584                                                                                                                             │
epoch:21, Classifier accuracy:0.7220000624656677                                                                                                                             │
epoch:22, Classifier accuracy:0.7660000324249268                                                                                                                             │
epoch:23, Classifier accuracy:0.7560000419616699                                                                                                                             │
epoch:24, Classifier accuracy:0.7350000143051147
epoch:25, Classifier accuracy:0.7200000286102295                                                                                                                             │Thu Nov 11 22:31:32 2021
epoch:26, Classifier accuracy:0.7480000257492065                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:27, Classifier accuracy:0.7930000424385071                                                                                                                             │| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
epoch:28, Classifier accuracy:0.7280000448226929                                                                                                                             │|-------------------------------+----------------------+----------------------+
epoch:29, Classifier accuracy:0.7560000419616699                                                                                                                             │| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
epoch:30, Classifier accuracy:0.7710000276565552                                                                                                                             │| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
epoch:31, Classifier accuracy:0.7190000414848328                                                                                                                             │|                               |                      |               MIG M. |
epoch:32, Classifier accuracy:0.7640000581741333                                                                                                                             │|===============================+======================+======================|
epoch:33, Classifier accuracy:0.7800000309944153                                                                                                                             │|   0  GeForce RTX 2070    Off  | 00000000:02:00.0 Off |                  N/A |
epoch:34, Classifier accuracy:0.781000018119812                                                                                                                              │| 66%   70C    P2   174W / 175W |   3783MiB /  7982MiB |     96%      Default |
epoch:35, Classifier accuracy:0.7730000615119934                                                                                                                             │|                               |                      |                  N/A |
epoch:36, Classifier accuracy:0.7710000276565552                                                                                                                             │+-------------------------------+----------------------+----------------------+
epoch:37, Classifier accuracy:0.7670000195503235                                                                                                                             │
epoch:38, Classifier accuracy:0.7520000338554382                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:39, Classifier accuracy:0.76500004529953                                                                                                                               │| Processes:                                                                  |
epoch:40, Classifier accuracy:0.781000018119812                                                                                                                              │|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
epoch:41, Classifier accuracy:0.7990000247955322                                                                                                                             │|        ID   ID                                                   Usage      |
epoch:42, Classifier accuracy:0.8050000667572021                                                                                                                             │|=============================================================================|
epoch:43, Classifier accuracy:0.7920000553131104                                                                                                                             │|    0   N/A  N/A      1358      G   /usr/lib/xorg/Xorg                134MiB |
epoch:44, Classifier accuracy:0.8060000538825989                                                                                                                             │|    0   N/A  N/A      2303      G   /usr/lib/xorg/Xorg                239MiB |
epoch:45, Classifier accuracy:0.8180000185966492                                                                                                                             │|    0   N/A  N/A      2471      G   /usr/bin/gnome-shell               10MiB |
epoch:46, Classifier accuracy:0.7940000295639038                                                                                                                             │|    0   N/A  N/A      2897      G   ...AAAAAAAAA= --shared-files        4MiB |
epoch:47, Classifier accuracy:0.7750000357627869                                                                                                                             │|    0   N/A  N/A   1347899      C   python3                          3359MiB |
epoch:48, Classifier accuracy:0.8270000219345093                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:49, Classifier accuracy:0.8220000267028809                                                                                                                             │
epoch:50, Classifier accuracy:0.815000057220459                                                                                                                              │
epoch:51, Classifier accuracy:0.7950000166893005                                                                                                                             │
epoch:52, Classifier accuracy:0.8050000667572021                                                                                                                             │
epoch:53, Classifier accuracy:0.7910000085830688                                                                                                                             │
epoch:54, Classifier accuracy:0.7950000166893005                                                                                                                             │
epoch:55, Classifier accuracy:0.800000011920929                                                                                                                              │
depoch:56, Classifier accuracy:0.7940000295639038                                                                                                                            │
epoch:57, Classifier accuracy:0.8190000653266907                                                                                                                             │
epoch:58, Classifier accuracy:0.8090000152587891                                                                                                                             │
epoch:59, Classifier accuracy:0.8030000329017639
epoch:60, Classifier accuracy:0.7750000357627869                                                                                                                             │Thu Nov 11 22:31:46 2021
epoch:61, Classifier accuracy:0.7600000500679016                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:62, Classifier accuracy:0.8080000281333923                                                                                                                             │| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
epoch:63, Classifier accuracy:0.7780000567436218                                                                                                                             │|-------------------------------+----------------------+----------------------+
epoch:64, Classifier accuracy:0.8290000557899475                                                                                                                             │| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
epoch:65, Classifier accuracy:0.7900000214576721                                                                                                                             │| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
epoch:66, Classifier accuracy:0.7630000114440918                                                                                                                             │|                               |                      |               MIG M. |
epoch:67, Classifier accuracy:0.7910000085830688                                                                                                                             │|===============================+======================+======================|
epoch:68, Classifier accuracy:0.8100000619888306                                                                                                                             │|   0  GeForce RTX 2070    Off  | 00000000:02:00.0 Off |                  N/A |
epoch:69, Classifier accuracy:0.7940000295639038                                                                                                                             │| 66%   71C    P2   166W / 175W |   3783MiB /  7982MiB |     96%      Default |
epoch:70, Classifier accuracy:0.7780000567436218                                                                                                                             │|                               |                      |                  N/A |
epoch:71, Classifier accuracy:0.8110000491142273                                                                                                                             │+-------------------------------+----------------------+----------------------+
epoch:72, Classifier accuracy:0.7980000376701355                                                                                                                             │
epoch:73, Classifier accuracy:0.831000030040741                                                                                                                              │+-----------------------------------------------------------------------------+
epoch:74, Classifier accuracy:0.8290000557899475                                                                                                                             │| Processes:                                                                  |
epoch:75, Classifier accuracy:0.7950000166893005                                                                                                                             │|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
epoch:76, Classifier accuracy:0.7910000085830688                                                                                                                             │|        ID   ID                                                   Usage      |
epoch:77, Classifier accuracy:0.8050000667572021                                                                                                                             │|=============================================================================|
epoch:78, Classifier accuracy:0.7950000166893005                                                                                                                             │|    0   N/A  N/A      1358      G   /usr/lib/xorg/Xorg                134MiB |
epoch:79, Classifier accuracy:0.8230000138282776                                                                                                                             │|    0   N/A  N/A      2303      G   /usr/lib/xorg/Xorg                239MiB |
epoch:80, Classifier accuracy:0.8350000381469727                                                                                                                             │|    0   N/A  N/A      2471      G   /usr/bin/gnome-shell               10MiB |
epoch:81, Classifier accuracy:0.7790000438690186                                                                                                                             │|    0   N/A  N/A      2897      G   ...AAAAAAAAA= --shared-files        3MiB |
epoch:82, Classifier accuracy:0.8070000410079956                                                                                                                             │|    0   N/A  N/A   1347899      C   python3                          3359MiB |
epoch:83, Classifier accuracy:0.7630000114440918                                                                                                                             │+-----------------------------------------------------------------------------+
epoch:84, Classifier accuracy:0.800000011920929                                                                                                                              │
epoch:85, Classifier accuracy:0.8020000457763672                                                                                                                             │
epoch:86, Classifier accuracy:0.8160000443458557                                                                                                                             │
epoch:87, Classifier accuracy:0.8010000586509705                                                                                                                             │
epoch:88, Classifier accuracy:0.7830000519752502                                                                                                                             │
epoch:89, Classifier accuracy:0.8480000495910645                                                                                                                             │
epoch:90, Classifier accuracy:0.7870000600814819                                                                                                                             │
epoch:91, Classifier accuracy:0.7690000534057617                                                                                                                             │
epoch:92, Classifier accuracy:0.8020000457763672                                                                                                                             │
epoch:93, Classifier accuracy:0.8350000381469727                                                                                                                             │
epoch:94, Classifier accuracy:0.8110000491142273
epoch:95, Classifier accuracy:0.7710000276565552                                                                                                                             │Thu Nov 11 22:31:58 2021
epoch:96, Classifier accuracy:0.796000063419342                                                                                                                              │+-----------------------------------------------------------------------------+
epoch:97, Classifier accuracy:0.831000030040741                                                                                                                              │| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
epoch:98, Classifier accuracy:0.7700000405311584                                                                                                                             │|-------------------------------+----------------------+----------------------+
epoch:99, Classifier accuracy:0.7940000295639038'''
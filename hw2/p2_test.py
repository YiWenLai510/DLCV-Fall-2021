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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")

parser.add_argument('--result_dir',default='p2_result')
parser.add_argument('--model_path',type=str)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

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
def calculate_acc(generator):
    # evaluate
    classifier = Classifier()
    
    classifier.load_state_dict(torch.load("Classifier.pth")['state_dict'])
    classifier = classifier.to(device)
    
    generator.eval()

    z = Variable(FloatTensor(np.random.normal(0, 1, (1000, opt.latent_dim))))
    labels = np.array([label for label in range(10) for j in range(100)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z.to(device), labels.to(device))

    cnt = 0

    for label in range(10):
        for j in range(100):
            idx = label * 100 + j
            img = gen_imgs.data[idx]
            # save_image(img, os.path.join(opt.result_dir,f"{label}_{j+1:03}.png" ),normalize=True) # need to resize to 5 * 5
            
    test_set = TestDataset('p2_result',transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
    
    for imgs, labels in testloader:
        labels = labels.to(device)
        logits = classifier(imgs.to(device))
        pred = torch.argmax(logits,dim=1)
        cnt += (pred == labels).sum()

    generator.train()    

    return   (cnt/1000)

generator = Generator().to(device)
generator.load_state_dict(torch.load(opt.model_path))
generator.eval()

z = Variable(FloatTensor(np.random.normal(0, 1, (1000, opt.latent_dim))))
labels = np.array([label for label in range(10) for j in range(100)])
labels = Variable(LongTensor(labels))
gen_imgs = generator(z.to(device), labels.to(device))
# print(gen_imgs.shape)
cnt = 0

for label in range(10):
    for j in range(100):
        idx = label * 100 + j
        img = gen_imgs.data[idx]
        save_image(img, os.path.join(opt.result_dir,f"{label}_{j+1:03}.png" ),normalize=True)

# print(calculate_acc(generator))

# plot
# z = Variable(FloatTensor(np.random.normal(0, 1, (100, opt.latent_dim))))
# labels = np.array([label for label in range(10) for j in range(10)])
# print(labels)
# labels = Variable(LongTensor(labels))
# gen_imgs = generator(z.to(device), labels.to(device))
# torchvision.utils.save_image(gen_imgs, 'p2_result.jpg', nrow=10)
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
from torch.utils.data.dataset import Dataset
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
import timm 
from timm.loss import LabelSmoothingCrossEntropy
from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer
from timm.scheduler.plateau_lr import PlateauLRScheduler
from pytorch_pretrained_vit import ViT

class myDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_lists = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):

        fileName = self.img_lists[idx]
        img_path = os.path.join(self.img_dir, fileName)
        image = Image.open(img_path).convert('RGB')
        # print(image.size)
        label = fileName[:-4].split('_')[0]

        if self.transform:
            image = self.transform(image)
        
        return  (image, int(label))


parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int, default=42)
parser.add_argument('--smoothLoss',type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training (default: 256)')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--save_dir',default='p1_models')
args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

np.random.seed(args.seed) # 42
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_tfm = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.1),
    # transforms.RandomPerspective(), # distortion_scale=0.2, p=0.5
    # transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15), # brightness=0.15, contrast=0.15, saturation=0.15
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),

])
test_tfm = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
train_dataset = myDataset(img_dir='hw3_data/p1_data/train',transform=train_tfm)
valid_dataset = myDataset(img_dir='hw3_data/p1_data/val',transform=test_tfm)
train_loader = DataLoader(train_dataset, args.batch_size,shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, args.batch_size,shuffle=True, num_workers=8)

# model_name = 'vit_base_patch16_384'
# model = timm.create_model(model_name, num_classes=37, pretrained=True, attn_drop_rate = 0.1)
model_name = 'B_16_imagenet1k'
model = ViT(model_name, num_classes=37, pretrained=True, attention_dropout_rate=0.1)
model.to(device)

if args.smoothLoss == 0:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = LabelSmoothingCrossEntropy(0.0)

# opt = SimpleNamespace()
# opt.weight_decay = 1e-1
# opt.lr = 1e-4
# opt.opt = 'adam' #'lookahead_adam' to use `lookahead`
# opt.momentum = 0.9

# optimizer = create_optimizer(opt, model)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.n_epochs*7) # number of update times

currentBest = -1
accumulation_steps = 128
step_cnt = 0

for epoch in range(args.n_epochs):

    model.train()
    t_cnt = 0
    for idx, batch in enumerate(train_loader):
        
        imgs, labels = batch
        logits = model(imgs.to(device))
        pred   = torch.argmax(logits, dim=1)
        
        loss   = criterion(logits, labels.to(device))
        loss   = loss / accumulation_steps
        
        loss.backward()

        if (idx+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_cnt += 1

        t_cnt += (pred == labels.to(device)).sum()
       
    print(f"[ Train | {epoch + 1:03d}/{args.n_epochs:03d} ] , acc = {t_cnt/len(train_dataset):.5f} ")

    model.eval()
    v_cnt = 0
    for batch in (valid_loader):

        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        pred   = torch.argmax(logits, dim=1)
        v_cnt += (pred == labels.to(device)).sum()

    print(f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] , acc = {v_cnt/len(valid_dataset):.5f} step_count: {step_cnt}")

    if v_cnt / len(valid_dataset) > currentBest:
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'{model_name}_{args.smoothLoss}_{args.seed}_{args.n_epochs}_noColor_normalize.pth'))
        currentBest = v_cnt/len(valid_dataset)
print(f'Best: {currentBest}')


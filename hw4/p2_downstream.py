import torch
from byol_pytorch import BYOL
import torchvision
from torchvision import models
import os
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',default='p2_downstream')
parser.add_argument('--pretrained', default='True')
parser.add_argument('--ta_backbone',default='True')
parser.add_argument('--classifier_only',default='False')
parser.add_argument('--n_epochs',type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-4,)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

NUM_CLASS = 65

class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        # print(self.data_df)
        self.data_df['tr_label'] = self.data_df['label'].rank(method='dense', ascending=True).astype(int)
        self.transform = transforms.Compose([
            # transforms.CenterCrop(256),
            transforms.Resize((128,128)),
            
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.1),
            # transforms.RandomPerspective(), # distortion_scale=0.2, p=0.5
            transforms.ColorJitter(),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.label = self.data_df['tr_label'].to_list()
        self.label = [x - 1 for x in self.label]
        # print(self.label)
        # print(self.data_df)

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        img_path = os.path.join(self.data_dir, path)
        label = self.label[index]
        image = self.transform(Image.open(img_path).convert('RGB'))
        # image = (Image.open(img_path).convert('RGB'))
        # return image, label
        return image, label

    def __len__(self):
        return len(self.data_df)

pretrained_backbone = models.resnet50(pretrained=False)


# for name, param in pretrained_backbone.named_parameters():
#    if param.requires_grad == True:
#         # params_to_update.append(param)
#     print("\t",name)
# print('before load')

if args.pretrained == 'True':
    
    improved_path = ''

    if args.ta_backbone == 'True':
        improved_path = 'hw4_data/pretrain_model_SL.pt'
    else:
        improved_path = 'p2_backbone/improved_resnet_backbone1.pt'
    
    pretrained_backbone.load_state_dict(torch.load(improved_path))


#     for name, param in pretrained_backbone.named_parameters():
#         if param.requires_grad == True:
#                 # params_to_update.append(param)
#             print("\t",name)
# print('before mod # of feats')

num_ftrs = pretrained_backbone.fc.in_features
pretrained_backbone.fc = nn.Linear(num_ftrs, NUM_CLASS)

if args.classifier_only == 'True':
    for name, param in pretrained_backbone.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        
params_to_update = []
for name,param in pretrained_backbone.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        # print("\t",name)

pretrained_backbone.to(device)

# print(pretrained_backbone)
total_params = sum(p.numel() for p in pretrained_backbone.parameters() if p.requires_grad)
print(f'number of params to be trained: {total_params}')



class_criterion = nn.CrossEntropyLoss()

train_dataset = MiniDataset('hw4_data/office/train.csv','hw4_data/office/train')
valid_dataset = MiniDataset('hw4_data/office/val.csv','hw4_data/office/val')
train_loader = DataLoader(train_dataset, batch_size=64 ,shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64 ,shuffle=True)


optimizer = torch.optim.Adam(params_to_update, lr=args.lr, betas=(args.b1, args.b2))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.n_epochs*len(train_loader))

t1_best = 0
for epoch in range(args.n_epochs):

    pretrained_backbone.train()
    
    tl = 0
    tr_cnt = 0

    for imgs, labels in (train_loader):
        
        imgs, labels = imgs.to(device), labels.to(device)
        # print(imgs.shape, labels.shape)
        class_logits = pretrained_backbone(imgs)

        optimizer.zero_grad()
        loss = class_criterion(class_logits, labels)
        
        tl += loss.item()
        pred = torch.argmax(class_logits, dim=1)
        tr_cnt += (pred == labels).sum()

        loss.backward()
        optimizer.step()
        
    print(f'Epoch: {epoch}, loss: {tl / len(train_loader)}, Acc: {tr_cnt / len(train_dataset)}')
    pretrained_backbone.eval()

    t1_cnt = 0

    for (t1_imgs, t1_labels) in (valid_loader):
        
        t1_labels  = t1_labels.to(device)
        logits1 = pretrained_backbone(t1_imgs.to(device))
        # print(logits1.shape)
        # print(t1_labels.shape)
        pred1 = torch.argmax(logits1, dim=1)
        # print(pred1.shape)
        t1_cnt += (pred1 == t1_labels).sum()

    
    print(f'Epoch: {epoch}, valAcc: {t1_cnt / len(valid_dataset)}')
    
    if  t1_cnt / len(valid_dataset) > t1_best :
        
        t1_best = t1_cnt / len(valid_dataset)
        torch.save(pretrained_backbone.state_dict(), os.path.join(args.save_dir, f'down_pretrained_{args.pretrained}_tabackbone_{args.ta_backbone}_classifieronly_{args.classifier_only}_nepoch_{args.n_epochs}.pth'))

    scheduler.step()
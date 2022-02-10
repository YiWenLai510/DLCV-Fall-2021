import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import csv
import random
import numpy as np
import pandas as pd

# from mini_imagenet import MiniImageNet
# from sampler import CategoriesSampler
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

ROOT_PATH = 'hw4_data/mini'
class MiniDataset(Dataset):

    def __init__(self, setname):
        csv_path = os.path.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            ids, name, wnid = l.split(',')
            path = os.path.join(ROOT_PATH, setname, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        # print(self.label)
        self.transform = transforms.Compose([
            # transforms.Resize(84),
            # transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch # no. of episodes
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # print(self.m_ind.shape)
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
                # print('l',l)
                # print(pos)
                # print(len(l[pos]), c)
            batch = torch.stack(batch).t().reshape(-1)
            
            yield batch
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )
    def forward(self,x):
        x = self.encoder(x)
        return x.view(x.size(0),-1)
    


def euclidean_metric(a, b):
    # print(a.shape)
    # print(b.shape)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    # print(a.shape)
    # print(b.shape)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_similarity(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    # logits = -((a - b)**2).sum(dim=2)
    logits = F.cosine_similarity(a, b, dim=2)
    # print(logits.shape)
    return logits

class Parametric_func(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(3200, 1600),
            nn.ReLU(),
            nn.Linear(1600, 100),
            nn.ReLU(),
            nn.Linear(100,1)
        )
         
        
    def forward(self, a, b): # a: torch.Size([150, 1600]), b: torch.Size([10, 1600])
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1) # torch.Size([150, 10, 1600])
        b = b.unsqueeze(0).expand(n, m, -1)
        c = torch.cat((a,b),2)
        # print(c.shape)
        ans = self.enc(c)
        # print(ans.shape)
        return ans.squeeze()

parser = argparse.ArgumentParser(description="Few shot learning")
parser.add_argument('--n_epochs', type=int, default=100)

parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--train-way', type=int, default=10)
parser.add_argument('--test-way', type=int, default=5)

parser.add_argument('--load', type=str, help="Model checkpoint path")
parser.add_argument('--test_csv', type=str, help="Testing images csv file")
parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
parser.add_argument('--testcase_csv', type=str, help="Test case csv")
parser.add_argument('--output_csv', type=str, help="Output filename")
parser.add_argument('--save_dir',default='save_models')

parser.add_argument('--distance',type=str, default='eu')

args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

model = Convnet().to(device) 

dis_func = Parametric_func().to(device)
dis_optim = torch.optim.Adam(dis_func.parameters(), lr=0.001)
# print(model) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_dataset = MiniDataset('train')
val_dataset   = MiniDataset('val')

train_sampler = CategoriesSampler(train_dataset.label, 100, args.train_way, args.shot + args.query)
val_sampler   = CategoriesSampler(val_dataset.label, 100, args.test_way, args.shot + args.query)

train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
val_loader   = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)

best_acc = 0

# print(train_loader)

for epoch in range(1, args.n_epochs + 1):

    lr_scheduler.step()

    model.train()

    tl = []
    ta = []

    for i, batch in enumerate(train_loader):

        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]

        # print(data_shot.shape)
        # print(data_query.shape)
        
        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

        label = torch.arange(args.train_way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        
        # print(i)
        if args.distance == 'eu':
            logits = euclidean_metric(model(data_query), proto)
        elif args.distance == 'cos':
            logits = cosine_similarity(model(data_query), proto)
        elif args.distance == 'param':
            logits  = dis_func(model(data_query), proto)
        # logits = euclidean_metric(model(data_query), proto)
        # logits = cosine_similarity(model(data_query), proto)
        # logits  = dis_func(model(data_query), proto)
        # print(logits.shape, label.shape)
        loss = F.cross_entropy(logits, label)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

        tl.append(loss.item())
        ta.append(acc)

        optimizer.zero_grad()
        if args.distance == 'param':
            dis_optim.zero_grad()

        loss.backward()
        optimizer.step()
        if args.distance == 'param':
            dis_optim.step()

    # print('epoch {},  loss={:.4f} acc={:.4f}'.format(epoch, sum(tl)/len(tl), sum(ta)/len(ta)))

    model.eval()
    vl = []
    va = []

    for i, batch in enumerate(val_loader):

        data, _ = [_.to(device) for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]

        # print(data_shot.shape)
        # print(data_query.shape)
        
        proto = model(data_shot)
        # print(proto.shape)
        proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

        label = torch.arange(args.test_way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        
        # print(label.shape)
        # print(i)
        if args.distance == 'eu':
            logits = euclidean_metric(model(data_query), proto)
        elif args.distance == 'cos':
            logits = cosine_similarity(model(data_query), proto)
        elif args.distance == 'param':
            logits  = dis_func(model(data_query), proto)

        loss = F.cross_entropy(logits, label)
        
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

        vl.append(loss.item())
        va.append(acc)
        # proto = None; logits = None; loss = None
        
    vas = sum(va) / len(va)
    if epoch == 100:
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, sum(vl)/len(vl), vas))

    if vas > best_acc:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_{args.distance}_shot_{args.shot}.pt') )
        if args.distance == 'param':
            torch.save(dis_func.state_dict(), os.path.join(args.save_dir, f'distance_param_shot_{args.shot}.pt') )
        best_acc = vas

# mini-Imagenet dataset
# class MiniDataset(Dataset):
#     def __init__(self, csv_path, data_dir):
#         self.data_dir = data_dir
#         self.data_df = pd.read_csv(csv_path).set_index("id")
#         # print(self.data_df)
#         self.data_df['tr_label'] = self.data_df['label'].rank(method='dense', ascending=True).astype(int)
#         self.transform = transforms.Compose([
#             # transforms.Resize(84),
#             # transforms.CenterCrop(84),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])

#         self.label = self.data_df['tr_label'].to_list()
#         self.label = [x - 1 for x in self.label]
#         # print(self.label)

#     def __getitem__(self, index):
#         path = self.data_df.loc[index, "filename"]
#         img_path = os.path.join(self.data_dir, path)
#         label = self.label[index]
#         image = self.transform(Image.open(img_path).convert('RGB'))
#         return image, label

#     def __len__(self):
#         return len(self.data_df)

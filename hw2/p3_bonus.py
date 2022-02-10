from typing import ForwardRef
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import argparse
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision
# from scipy.stats import entropy
from tqdm.auto import tqdm

writer = SummaryWriter()
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
class DigitDataset(nn.Module):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path,sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')
        self.images_folder = images_folder
        self.transform = transform        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.iloc[index]['image_name']
        label = self.df.iloc[index]['label']
        image = Image.open(os.path.join(self.images_folder, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# class ResNetSimCLR(nn.Module):
#     def __init__(self, base_model, out_dim):
#         super(ResNetSimCLR, self).__init__()
#         self.backbone = torchvision.models.resnet18(pretrained=False, num_classes=out_dim)
#         self.buffersdim_mlp = self.backbone.fc.in_features
#         self.backbone.fc = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp), nn.ReLU(), self.backbone.fc)
#     def forward(self, x):
#         return self.backbone(x)
class Feature_extractor(nn.Module):
    def __init__(self) -> None:
        super(Feature_extractor, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, 3, 1, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 256, 3, 1, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    def forward(self, x):
        batchsize = x.shape[0]
        x = self.conv(x)
        # print(x.shape)
        x = x.view(batchsize,-1)
        # print('feat out', x.shape)
        return x

class Label_predictor(nn.Module):
    def __init__(self) -> None:
        super(Label_predictor,self).__init__()
        self.in_dim = 128
        self.layer = nn.Sequential(
            nn.Linear(256, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, 10),
        )
    def forward(self, x):
        x = self.layer(x)

        # print('label', x.shape)

        return x

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.in_dim = 128
        self.layer = nn.Sequential(
            nn.Linear(256, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, 10),
        )
    def forward(self, x):
        # print(x.shape)
        batchsize = x.shape[0]
        x = self.conv(x)
        # print(x.shape)
        x = x.view(batchsize,-1)
        # print(x.shape)
        x = self.layer(x)
        return x
class prev_dataset(nn.Module):
    def __init__(self, labels, images_folder, transform = None):
        
        self.images_folder = images_folder
        self.transform = transform    
        self.label_list = labels
        self.img_list = sorted(os.listdir(self.images_folder))    
        # print(self.label_list[0])
        # print(self.label_list[2])
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        filename = self.img_list[index]
        image = Image.open(os.path.join(self.images_folder, filename)).convert('RGB')
        label = self.label_list[index]
        # label = self.classifier(image.to(device))
        if self.transform is not None:
            image = self.transform(image)
        # print(index, filename, label)
        return image, label
def train_bonus(args, source, target):

    digits_homedir = 'hw2_data/digits'

    source_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    target_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    source_dataset = DigitDataset(  csv_path= os.path.join(digits_homedir, source, 'train.csv'),
                                    images_folder= os.path.join(digits_homedir, source, 'train'), 
                                    transform= source_transform,)
    target_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target, 'test'),
                                    transform= target_transform)
    test_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target, 'test'),
                                    transform= target_transform)

    source_loader = DataLoader(source_dataset, batch_size=256, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('Training backbone.......')

    feature_extractor = Feature_extractor().to(device)
    label_predictor = Label_predictor().to(device)
    class_criterion = nn.CrossEntropyLoss()

    # checkpoint = torch.load(os.path.join(args.save_bonus, f'base_S_{source}_T_{target}.pth'))
    
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # backbone.load_state_dict(checkpoint['model_state_dict'])
    # backbone.load_state_dict(torch.load(f'Source_models/single_{source}_45.pth'))
    # for k, v in backbone.named_parameters():
    #     v.requires_grad = True
    if source != 'svhn':
        param_group = []
        for k, v in feature_extractor.named_parameters():
            param_group += [{'params':v}]
        for k, v in label_predictor.named_parameters():
            param_group += [{'params':v}]

        optimizer = optim.Adam(param_group,lr=args.lr, betas=(args.b1, args.b2))
        best = 0
        for epoch in range(30):

            feature_extractor.train()
            label_predictor.train()

            for imgs, labels  in (source_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                class_logits = label_predictor(feature_extractor(imgs))
                optimizer.zero_grad()
                loss = class_criterion(class_logits, labels)
                loss.backward()
                optimizer.step()

            feature_extractor.eval()
            label_predictor.eval()

            t_cnt = 0

            for t_imgs, t_labels in target_loader:
                logits = label_predictor(feature_extractor(t_imgs.to(device)))
                pred = torch.argmax(logits, dim=1)
                # print(logits.shape)
                # print(pred.shape)
                # print(t_labels.shape)
                t_cnt += ( pred == t_labels.to(device)).sum()

            print(f'Epoch:{epoch}, Loss:{loss},  Source:{source}, Target:{target}, Acc:{t_cnt / len(test_dataset)}')
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': backbone.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss,
            #     }, os.path.join(args.save_bonus, f'base_S_{source}_T_{target}.pth'))
            if t_cnt / len(test_dataset) > best:
                best = t_cnt / len(test_dataset)
                torch.save(feature_extractor.state_dict(),os.path.join(args.save_bonus,f'base_feat_S_{source}_T_{target}.pth') )
                torch.save(label_predictor.state_dict(),os.path.join(args.save_bonus,f'base_label_S_{source}_T_{target}.pth') )

    print(f'Training UDA {source} to {target}...............')

    
    feature_extractor = Feature_extractor().to(device)
    label_predictor = Label_predictor().to(device)

    feature_extractor.load_state_dict(torch.load(f'p3_bonus_save_models/base_feat_S_{source}_T_{target}.pth'))
    label_predictor.load_state_dict(torch.load(f'p3_bonus_save_models/base_label_S_{source}_T_{target}.pth'))

    criterion = nn.CrossEntropyLoss()
    param_group = []
    for k, v in feature_extractor.named_parameters():
        param_group += [{'params':v}]
    for k, v in label_predictor.named_parameters():
        v.requires_grad = False
    optimizer = optim.Adam(param_group ,lr=args.lr, betas=(args.b1, args.b2))

    bestuda= 0

    for epoch in range(args.n_epochs):

        feature_extractor.eval()
        label_predictor.eval()

        labels = []
        t_cnt = 0

        for t_imgs, t_labels in test_loader:
            logits = label_predictor(feature_extractor(t_imgs.to(device)))
            labels.append(logits.detach().cpu())
            pred = torch.argmax(logits, dim=1)
            t_cnt += ( pred == t_labels.to(device)).sum()
        print(f'Previous epoch: {epoch-1}',t_cnt / len(test_dataset))

        labels = torch.cat(labels, dim=0)
       
        mem_dataset = prev_dataset(labels=labels, images_folder=os.path.join(digits_homedir, target, 'test')
                                    , transform= target_transform)
        mem_loader = DataLoader(mem_dataset, batch_size=args.batch_size, shuffle=True)

        feature_extractor.train()
        label_predictor.train()
        
        for idx, data in enumerate(mem_loader):
            
            target_data, target_label = data
            target_data, target_label = target_data.to(device), target_label.to(device)
            logits = label_predictor(feature_extractor(target_data))
          
            classifier_loss = 0.3 * criterion(logits, target_label)

            softmax_out = nn.Softmax(dim=1)(logits)
            entropy_loss = torch.mean(Entropy(softmax_out))

            msoftmax = softmax_out.mean(dim=0)
            entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * 1.0
            classifier_loss += im_loss

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
            writer.add_scalar("classifier_loss", classifier_loss, epoch)
        # evaluate
        feature_extractor.eval()
        label_predictor.eval()

        t_cnt = 0

        for t_imgs, t_labels in target_loader:
            logits = label_predictor(feature_extractor(t_imgs.to(device)))
            pred = torch.argmax(logits, dim=1)
            t_cnt += ( pred == t_labels.to(device)).sum()

        print(f'Epoch:{epoch},Loss:{classifier_loss},  Source:{source}, Target:{target}, Acc:{t_cnt / len(test_dataset)}')
        if t_cnt / len(test_dataset) > bestuda:
            bestuda = t_cnt / len(test_dataset)
            torch.save(feature_extractor.state_dict(),os.path.join(args.save_bonus,f'bonus_feat_S_{source}_T_{target}.pth') )
            torch.save(label_predictor.state_dict(),os.path.join(args.save_bonus,f'bonus_label_S_{source}_T_{target}.pth') )
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': backbone.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': classifier_loss,
        #     }, os.path.join(args.save_bonus, f'bonus_S_{source}_T_{target}.pth'))
writer.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,)
parser.add_argument('--n_epochs', type=int, default=100,)
parser.add_argument('--lr', type=float, default=2e-4,)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--save_bonus', default='p3_bonus_save_models')
parser.add_argument('--uda',type=str,default='simclr')
args = parser.parse_args()
print(args)
os.makedirs(args.save_bonus, exist_ok=True)

dataSource = ['svhn', 'mnistm', 'usps']

for idx, source in enumerate(dataSource):
    if idx == 2:
        train_bonus(args, source, dataSource[0])
    else:
        train_bonus(args, source, dataSource[idx+1])

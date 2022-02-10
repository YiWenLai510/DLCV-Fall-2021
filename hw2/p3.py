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

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        image = Image.open(os.path.join(self.images_folder, filename)).convert('RGB')#.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
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

class u2s_FeatureExtractor(nn.Module):

    def __init__(self):
        super(u2s_FeatureExtractor, self).__init__()

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
        # print(x.shape)
        batchsize = x.shape[0]
        x = self.conv(x)
        # print(x.shape)
        x = x.view(batchsize,-1)
        # print(x.shape)
        return x
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 2, 1), #256*12*12
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 2, 1), #256*10*10
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 2, 1), #512*8*8
            nn.BatchNorm2d(512),
            nn.ReLU(),

        )
        
    def forward(self, x):
        batchsize = x.shape[0]
        x = self.conv(x)
        # print(x.shape)
        x = x.squeeze()
        return x

class u2s_LabelPredictor(nn.Module):

    def __init__(self):
        super(u2s_LabelPredictor, self).__init__()
        self.in_dim = 128
        self.layer = nn.Sequential(
            nn.Linear(256, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, 10),
        )

    def forward(self, h):
        # print(h.shape)
        # batchsize = h.shape[0]
        c = self.layer(h)
        # print(c.shape)
        # c = c
        return c
class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.in_dim = 512
        self.layer = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c
class u2s_DomainClassifier(nn.Module):

    def __init__(self):
        super(u2s_DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.in_dim = 512
        self.layer = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(),

            nn.Linear(self.in_dim, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,)
parser.add_argument('--n_epochs', type=int, default=100,)
parser.add_argument('--lr', type=float, default=2e-4,)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--save_single',default='p3_single_save_models')
parser.add_argument('--save_adaptive', default='p3_adaptive_save_models')
# parser.add_argument('--log_dir',default='p3_log')
# parser.add_argument('--result_dir',default='p3_result')
args = parser.parse_args()
print(args)

os.makedirs(args.save_adaptive, exist_ok=True)
# os.makedirs(args.log_dir, exist_ok=True)
# os.makedirs(args.result_dir, exist_ok=True)


source_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
target_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
adaptive_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def train_single(source, target1, target2):

    print(source, target1, target2)
    digits_homedir = 'hw2_data/digits'


    source_dataset = DigitDataset(  csv_path= os.path.join(digits_homedir, source, 'train.csv'),
                                    images_folder= os.path.join(digits_homedir, source, 'train'), 
                                    transform= source_transform, gray = (source == 'usps'))

    target1_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target1, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target1, 'test'),
                                    transform= target_transform, gray = (target1 == 'usps'))

    target2_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target2, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target2, 'test'), 
                                    transform= target_transform, gray = (target2 == 'usps'))
    
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    target1_loader = DataLoader(target1_dataset, batch_size=args.batch_size, shuffle=True)
    target2_loader = DataLoader(target2_dataset, batch_size=args.batch_size, shuffle=True)

    classifier = Classifier().to(device)

    class_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(classifier.parameters(),lr=args.lr, betas=(args.b1, args.b2))

    t1_best, t2_best = 0, 0

    for epoch in range(args.n_epochs):

        classifier.train()

        for imgs, labels in (source_loader):
            
            imgs, labels = imgs.to(device), labels.to(device)
            # print(imgs.shape, labels.shape)
            class_logits = classifier(imgs)
            
            optimizer.zero_grad()
            
            loss = class_criterion(class_logits, labels)
            loss.backward()
            optimizer.step()
            

        classifier.eval()

        t1_cnt = 0
        t2_cnt = 0

        for (t1_imgs, t1_labels) in (target1_loader):
            
            t1_labels  = t1_labels.to(device)
            logits1 = classifier(t1_imgs.to(device))
            # print(logits1.shape)
            # print(t1_labels.shape)
            pred1 = torch.argmax(logits1, dim=1)
            # print(pred1.shape)
            t1_cnt += (pred1 == t1_labels).sum()

        for t2_imgs, t2_labels in target2_loader:
            t2_labels = t2_labels.to(device)
            logits2 = classifier(t2_imgs.to(device))
            pred2 = torch.argmax(logits2, dim=1)
            t2_cnt += ( pred2== t2_labels).sum()

        if epoch == 0 or (epoch+1) % 5 == 0:
            print(f'Epoch:{epoch}, Source:{source}, Target1:{target1}, Acc:{t1_cnt / len(target1_dataset)}, Target2:{target2}, Acc:{t2_cnt / len(target2_dataset)}')
        
        if  t1_cnt / len(target1_dataset) > t1_best :
            t1_best = t1_cnt / len(target1_dataset)
        if  t2_cnt / len(target2_dataset) > t2_best : 
            t2_best = t2_cnt / len(target2_dataset)

        torch.save(classifier.state_dict(), os.path.join(args.save_single, f'single_{source}_{epoch}.pth'))

    print(f' Source:{source}, Target:{target1}, Acc:{t1_best}')
    print(f' Source:{source}, Target:{target2}, Acc:{t2_best}')
def train_adaption(source, target):
    
    print(source, target)
    
    digits_homedir = 'hw2_data/digits'

    source_dataset = DigitDataset(  csv_path= os.path.join(digits_homedir, source, 'train.csv'),
                                    images_folder= os.path.join(digits_homedir, source, 'train'), 
                                    transform= adaptive_transform)
    target_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target, 'test'),
                                    transform= adaptive_transform)
    test_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target, 'test'),
                                    transform= target_transform)

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters(),lr=args.lr, betas=(args.b1, args.b2))
    optimizer_C = optim.Adam(label_predictor.parameters(),lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(domain_classifier.parameters(),lr=args.lr, betas=(args.b1, args.b2))

    best_acc, best_epoch = 0, 0

    for epoch in range(args.n_epochs):
        
        feature_extractor.train()
        label_predictor.train()

        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)) :
            
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
            # set domain label of source data to be 1.
            domain_label[:source_data.shape[0]] = 1
            
            feature = feature_extractor(mixed_data.to(device))
            domain_logits = domain_classifier(feature)
            loss = domain_criterion(domain_logits, domain_label) 
            
            optimizer_D.zero_grad()
            loss.backward(retain_graph = True)
            optimizer_D.step()

            class_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            loss2 = class_criterion(class_logits, source_label.to(device)) - 0.1 * domain_criterion(domain_logits, domain_label)
            
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()
            loss2.backward()
            optimizer_F.step()
            optimizer_C.step()


        feature_extractor.eval()
        label_predictor.eval()

        t_cnt = 0
        for t_imgs, t_labels in test_loader:
            feats  = feature_extractor(t_imgs.to(device))
            logits = label_predictor(feats)
            pred = torch.argmax(logits, dim=1)
            t_cnt += ( pred == t_labels.to(device)).sum()
        print(f'Epoch:{epoch}, Source:{source}, Target:{target}, Acc:{t_cnt / len(test_dataset)}')
        
        if (t_cnt / len(target_dataset)) > best_acc:
            best_acc = t_cnt / len(target_dataset)
            best_epoch = epoch
            torch.save(feature_extractor.state_dict(), os.path.join(args.save_adaptive, f'Modfeature_S_{source}_T_{target}.pth'))
            torch.save(label_predictor.state_dict(), os.path.join(args.save_adaptive, f'Modlabel_S_{source}_T_{target}.pth'))

    print(f'Source: {source}, Target: {target}, BestAcc: {best_acc}, Epoch: {best_epoch}')



dataSource = ['svhn', 'mnistm', 'usps']
# single domain
# print('Train single')
# for idx, source in enumerate(dataSource):
#     if idx == 2:
#         train_single(source, source, dataSource[0])
#     else:
#         train_single(source, source, dataSource[idx+1])

print('Train Domain Adaption')
for idx, source in enumerate(dataSource):
    if idx == 2:
        train_adaption(source, dataSource[0])
    else:
        train_adaption(source, dataSource[idx+1])








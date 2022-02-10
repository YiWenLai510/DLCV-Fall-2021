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
def train_adaption(args, source, target):

    print(f'Improved from adaption from {source} to {target}')

    digits_homedir = 'hw2_data/digits'

    target_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    target_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target, 'test'),
                                    transform= target_transform)
    test_dataset = DigitDataset( csv_path= os.path.join(digits_homedir, target, 'test.csv'),
                                    images_folder= os.path.join(digits_homedir, target, 'test'),
                                    transform= target_transform)

    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)

    if source == 'usps': 
        feature_extractor = u2s_FeatureExtractor().to(device)
        label_predictor = u2s_LabelPredictor().to(device)

    model_path_feats = f'p3_save_improved_adaptive/bonus_feat_S_{source}_T_{target}.pth'
    model_path_labels = f'p3_save_improved_adaptive/bonus_label_S_{source}_T_{target}.pth'

    feature_extractor.load_state_dict(torch.load(model_path_feats))
    label_predictor.load_state_dict(torch.load(model_path_labels))
    feature_extractor.eval()
    label_predictor.eval()

    t_cnt = 0
    for t_imgs, t_labels in test_loader:
        feats  = feature_extractor(t_imgs.to(device))
        logits = label_predictor(feats)
        pred = torch.argmax(logits, dim=1)
        t_cnt += ( pred == t_labels.to(device)).sum()
    print('Showing previous result.......', t_cnt / len(test_dataset))
    
    # criterion = nn.CrossEntropyLoss()
    # param_group = []
    # for k, v in feature_extractor.named_parameters():
    #     param_group += [{'params':v}]
    # for k, v in label_predictor.named_parameters():
    #     v.requires_grad = False
    # optimizer = optim.Adam(param_group ,lr=args.lr, betas=(args.b1, args.b2))

    # bestuda = 0

    # for epoch in range(args.n_epochs):

    #     feature_extractor.eval()
    #     label_predictor.eval()

    #     labels = []

    #     for t_imgs, t_labels in test_loader:
    #         logits = label_predictor(feature_extractor(t_imgs.to(device)))
    #         labels.append(logits.detach().cpu())

    #     labels = torch.cat(labels, dim=0)
       
    #     mem_dataset = prev_dataset(labels=labels, images_folder=os.path.join(digits_homedir, target, 'test')
    #                                 , transform= target_transform)
    #     mem_loader = DataLoader(mem_dataset, batch_size=args.batch_size, shuffle=True)

    #     feature_extractor.train()
    #     label_predictor.train()
        
    #     for idx, data in enumerate(mem_loader):
            
    #         target_data, target_label = data
    #         target_data, target_label = target_data.to(device), target_label.to(device)
    #         logits = label_predictor(feature_extractor(target_data))
          
    #         classifier_loss = 0.3 * criterion(logits, target_label)

    #         softmax_out = nn.Softmax(dim=1)(logits)
    #         entropy_loss = torch.mean(Entropy(softmax_out))

    #         msoftmax = softmax_out.mean(dim=0)
    #         entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    #         im_loss = entropy_loss * 1.0
    #         classifier_loss += im_loss

    #         optimizer.zero_grad()
    #         classifier_loss.backward()
    #         optimizer.step()
    #     # evaluate
    #     feature_extractor.eval()
    #     label_predictor.eval()

    #     t_cnt = 0

    #     for t_imgs, t_labels in target_loader:
    #         logits = label_predictor(feature_extractor(t_imgs.to(device)))
    #         pred = torch.argmax(logits, dim=1)
    #         t_cnt += ( pred == t_labels.to(device)).sum()

    #     print(f'Epoch:{epoch},Loss:{classifier_loss},  Source:{source}, Target:{target}, Acc:{t_cnt / len(test_dataset)}')
    #     if t_cnt / len(test_dataset) > bestuda:
    #         bestuda = t_cnt / len(test_dataset)
    #         torch.save(feature_extractor.state_dict(),os.path.join(args.save_improved_adaptive,f'bonus_feat_S_{source}_T_{target}.pth') )
    #         torch.save(label_predictor.state_dict(),os.path.join(args.save_improved_adaptive,f'bonus_label_S_{source}_T_{target}.pth') )


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,)
parser.add_argument('--n_epochs', type=int, default=150,)
parser.add_argument('--lr', type=float, default=2e-4,)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--save_improved_adaptive', default='p3_save_improved_adaptive')
# parser.add_argument('--log_dir',default='p3_log')
# parser.add_argument('--result_dir',default='p3_result')
args = parser.parse_args()
print(args)

os.makedirs(args.save_improved_adaptive, exist_ok=True)


dataSource = ['svhn', 'mnistm', 'usps']

print('Train imporved Domain Adaption')
for idx, source in enumerate(dataSource):
    if idx == 2:
        train_adaption(args, source, dataSource[0])
    else:
        train_adaption(args, source, dataSource[idx+1])

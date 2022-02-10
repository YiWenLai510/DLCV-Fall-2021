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
        batchsize = x.shape[0]
        x = self.conv(x)
        x = x.view(batchsize,-1)
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
        c = self.layer(h)
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

class DigitDataset(nn.Module):
    def __init__(self, images_folder, transform = None):
        # self.df = pd.read_csv(csv_path,sep=r'\s*,\s*',
        #                    header=0, encoding='ascii', engine='python')
        self.images_folder = images_folder
        self.transform = transform
        self.img_lists = sorted(os.listdir(self.images_folder))
        # self.class2index = {"cat":0, "dog":1}
        # print(self.df.iloc[0]['image_name'])        
        # print(self.df.iloc[0]['label'])        
    def __len__(self):
        return len(self.img_lists)
    def __getitem__(self, index):
        filename = self.img_lists[index]
        # label = self.df.iloc[index]['label']
        image = Image.open(os.path.join(self.images_folder, filename)).convert('RGB')#.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, filename

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--image_folder',type=str)
parser.add_argument('--target_domain',type=str)
parser.add_argument('--output_file',type=str)
# parser.add_argument('--model_path_feats',type=str)
# parser.add_argument('--model_path_labels',type=str)

args = parser.parse_args()
print(args)

target_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

if args.target_domain == 'svhn':

    feature_extractor = u2s_FeatureExtractor().to(device)
    label_predictor = u2s_LabelPredictor().to(device)

    test_dataset = DigitDataset(images_folder=args.image_folder, transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # model_path_feats = 'final/feature_S_usps_T_svhn.pth'
    # model_path_labels = 'final/label_S_usps_T_svhn.pth'
    model_path_feats = f'hw2_p3_{args.target_domain}_feature.pt'
    model_path_labels = f'hw2_p3_{args.target_domain}_label.pt'

    feature_extractor.load_state_dict(torch.load(model_path_feats))
    label_predictor.load_state_dict(torch.load(model_path_labels))

    feature_extractor.eval()
    label_predictor.eval()

    predictions = []
    fileNames = []

    # t_cnt = 0
    for t_imgs,  files in test_loader:
        feats  = feature_extractor(t_imgs.to(device))
        logits = label_predictor(feats)
        pred = torch.argmax(logits, dim=1)
        # t_cnt += ( pred == t_labels.to(device)).sum()
        predictions.extend(pred.cpu().numpy().tolist())
        # print(fileNames)
        fileNames.extend(files)
    # print(f'target:{args.target_domain} Acc:{t_cnt / len(test_dataset)}')

    with open(args.output_file, "w") as f:
        f.write("image_name,label\n")
        for i, pred in  enumerate(predictions):
            f.write(f"{fileNames[i]},{pred}\n")
else:

    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)

    test_dataset = DigitDataset(images_folder=args.image_folder, transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_path_feats = f'hw2_p3_{args.target_domain}_feature.pt'
    model_path_labels = f'hw2_p3_{args.target_domain}_label.pt'

    # if args.target_domain == 'usps':
    #     model_path_feats = 'p3_adaptive_save_models/Modfeature_S_mnistm_T_usps.pth'
    #     model_path_labels = 'p3_adaptive_save_models/Modlabel_S_mnistm_T_usps.pth'
    # if args.target_domain == 'mnistm':
    #     model_path_feats = 'p3_adaptive_save_models/Modfeature_S_svhn_T_mnistm.pth'
    #     model_path_labels = 'p3_adaptive_save_models/Modlabel_S_svhn_T_mnistm.pth'

    feature_extractor.load_state_dict(torch.load(model_path_feats))
    label_predictor.load_state_dict(torch.load(model_path_labels))

    feature_extractor.eval()
    label_predictor.eval()

    predictions = []
    fileNames = []

    t_cnt = 0
    for t_imgs, files in test_loader:
        feats  = feature_extractor(t_imgs.to(device))
        logits = label_predictor(feats)
        pred = torch.argmax(logits, dim=1)
        # t_cnt += ( pred == t_labels.to(device)).sum()
        predictions.extend(pred.cpu().numpy().tolist())
        fileNames.extend(files)
    # print(f'target:{args.target_domain} Acc:{t_cnt / len(test_dataset)}')

    with open(args.output_file, "w") as f:
        f.write("image_name,label\n")
        for i, pred in  enumerate(predictions):
            f.write(f"{fileNames[i]},{pred}\n")
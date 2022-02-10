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
# from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import json


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--test_csv', type=str)
parser.add_argument('--test_dir', type=str)
parser.add_argument('--out_csv', type=str)

args = parser.parse_args()
print(args)

NUM_CLASS = 65

class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        # print(self.data_df)
        # self.data_df['tr_label'] = self.data_df['label'].rank(method='dense', ascending=True).astype(int)
        self.transform = transforms.Compose([
            # transforms.CenterCrop(256),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # self.label = self.data_df['tr_label'].to_list()
        # self.label = [x - 1 for x in self.label]
        # print(self.label)
        # print(self.data_df)
        # self.data_df['tr_label'] -= 1
        # area_dict = dict(zip(self.data_df.tr_label, self.data_df.label))
        # print(area_dict)
        # with open('label_mapping.json', 'w') as fp:
            # json.dump(area_dict, fp)

        # print(self.data_df)

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        img_path = os.path.join(self.data_dir, path)
        # label = self.label[index]
        image = self.transform(Image.open(img_path).convert('RGB'))
        # image = (Image.open(img_path).convert('RGB'))
        # return image, label
        # return image, label
        return image, path

    def __len__(self):
        return len(self.data_df)


pretrained_backbone = models.resnet50(pretrained=False)
num_ftrs = pretrained_backbone.fc.in_features
pretrained_backbone.fc = nn.Linear(num_ftrs, NUM_CLASS)
pretrained_backbone.load_state_dict(torch.load('hw4_p2_model.pt'))
# pretrained_backbone.load_state_dict(torch.load('p2_downstream/down_pretrained_True_tabackbone_False_classifieronly_False.pth'))

pretrained_backbone.to(device)
valid_dataset = MiniDataset(args.test_csv,args.test_dir)
valid_loader = DataLoader(valid_dataset, batch_size=64 ,shuffle=False)

predictions = []
fileNames = []

# t1_cnt = 0

# for (t1_imgs, t1_labels) in (valid_loader):
    
#     t1_labels  = t1_labels.to(device)
#     logits1 = pretrained_backbone(t1_imgs.to(device))
#     # print(logits1.shape)
#     # print(t1_labels.shape)
#     pred1 = torch.argmax(logits1, dim=1)
#     # print(pred1.shape)
#     t1_cnt += (pred1 == t1_labels).sum()


# print(f'valAcc: {t1_cnt / len(valid_dataset)}')

for (t1_imgs, filepaths) in (valid_loader):
    
    # t1_labels  = t1_labels.to(device)
    logits1 = pretrained_backbone(t1_imgs.to(device))
    # print(logits1.shape)
    # print(t1_labels.shape)
    pred1 = torch.argmax(logits1, dim=1)
    predictions.extend(pred1.cpu().numpy().tolist())
    fileNames.extend(filepaths)

with open('label_mapping.json', 'r') as fp:
    label_dict = json.load(fp)
# print(label_dict)
with open(args.out_csv, "w") as f:
    # The first row must be "Id, Category"
    f.write("id,filename,label\n")
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{fileNames[i]},{label_dict[str(pred)]}\n")
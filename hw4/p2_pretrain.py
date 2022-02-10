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
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',default='p2_backbone')
args = parser.parse_args()
print(args)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

os.makedirs(args.save_dir, exist_ok=True)

class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        # print(self.data_df)
        # self.data_df['tr_label'] = self.data_df['label'].rank(method='dense', ascending=True).astype(int)
        self.transform = transforms.Compose([
            # transforms.Resize(84),
            # transforms.CenterCrop(84),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # self.label = self.data_df['tr_label'].to_list()
        # self.label = [x - 1 for x in self.label]
        # print(self.label)

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        img_path = os.path.join(self.data_dir, path)
        # label = self.label[index]
        image = self.transform(Image.open(img_path).convert('RGB'))
        # image = (Image.open(img_path).convert('RGB'))
        # return image, label
        return image

    def __len__(self):
        return len(self.data_df)

backbone = models.resnet50(pretrained=False)
# backbone.load_state_dict(torch.load('hw4_data/pretrain_model_SL.pt'))
backbone.to(device)
learner = BYOL(
    backbone,
    image_size = 128,
    hidden_layer = 'avgpool'
)
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

train_dataset = MiniDataset('hw4_data/mini/train.csv','hw4_data/mini/train')
loader = DataLoader(train_dataset, 64 ,shuffle=True, num_workers=8)

for i in range(50):
    # images = sample_unlabelled_images()
    ls = 0
    for batch in tqdm(loader):
        loss = learner(batch.to(device))
        
        ls += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
    print(f'iter: {i}, loss: {ls/len(loader)}')
    torch.save(backbone.state_dict(), os.path.join(args.save_dir, 'improved_resnet_backbone1.pt'))

# save your improved network
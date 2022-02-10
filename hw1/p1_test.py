import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import os
import argparse
from torch.utils.data.dataset import Dataset


class myDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_lists = sorted(os.listdir(self.img_dir))
        # print(self.img_lists)
    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        fileName = self.img_lists[idx]
        img_path = os.path.join(self.img_dir, fileName)
        image = Image.open(img_path)
        # label = fileName[:-4].split('_')[0]

        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = (self.target_transform(label))
        return fileName, image #, label
        # return  (fileName, image, int(label))

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--model_path',type=str)
args = parser.parse_args()
print(args)


device = "cuda" if torch.cuda.is_available() else "cpu"

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_dataset = myDataset(img_dir=args.img_dir,transform=test_tfm)
dataloader = DataLoader(test_dataset, batch_size=64,shuffle=False, num_workers=0)

outfile_dir = args.outfile

model_path = args.model_path

model = torchvision.models.vgg13_bn(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs,50)

model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

predictions = []
fileNames = []

for batch in (dataloader):
    imgNames, imgs = batch
    with torch.no_grad():
        logits = model(imgs.to(device))

    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    fileNames.extend(imgNames)

with open(outfile_dir, "w") as f:
    # The first row must be "Id, Category"
    f.write("image_id,label\n")
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{fileNames[i]},{pred}\n")
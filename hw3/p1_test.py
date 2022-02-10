import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import timm 
from torch.utils.data.dataset import Dataset
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--id',default='')
parser.add_argument('--seed',type=int, default=42)
parser.add_argument('--img_folder',type=str)
parser.add_argument('--output_file',type=str)
parser.add_argument('--model_path',type=str)
args = parser.parse_args()
# print(args)
class myDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_lists = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):

        fileName = self.img_lists[idx]
        img_path = os.path.join(self.img_dir, fileName)
        image = Image.open(img_path).convert('RGB')
        # print(image.size)
        # label = fileName[:-4].split('_')[0]

        if self.transform:
            image = self.transform(image)
        
        return  image, fileName # , int(label)

np.random.seed(args.seed) # 42
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

test_tfm = transforms.Compose([
    transforms.CenterCrop(384),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
valid_dataset = myDataset(img_dir=args.img_folder,transform=test_tfm)
valid_loader = DataLoader(valid_dataset, 64, shuffle=False, num_workers=8)

model_name = 'vit_base_patch16_384'
model = timm.create_model(model_name, num_classes=37, pretrained=True, attn_drop_rate = 0.1)
model.to(device)
# print(model)
# model_path = f'p1_models/vit_base_patch16_384_{args.id}.pth'
# model_path =  #f'p1_models/vit_base_patch16_384_0_200000_noColor.pth' # best

model.load_state_dict(torch.load(args.model_path))
model.eval()

v_cnt = 0
predictions = []
fileNames = []

for batch in (valid_loader):

    imgs, imgids = batch # , labels
    with torch.no_grad():
        logits = model(imgs.to(device))
    pred = torch.argmax(logits, dim=1)
    # v_cnt += (pred == labels.to(device)).sum()

    predictions.extend(pred.cpu().numpy().tolist())
        # print(fileNames)
    fileNames.extend(imgids)

# print(f"path = {model_path}, acc = {v_cnt/len(valid_dataset):.5f} ")

with open(args.output_file, "w") as f:
        f.write("image_name,label\n")
        for i, pred in  enumerate(predictions):
            f.write(f"{fileNames[i]},{pred}\n")



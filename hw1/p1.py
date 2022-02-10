import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import os
import cv2
from tqdm.auto import tqdm
import argparse
from torch.utils.data.dataset import Dataset

# def CreateImageFolder(folder, type):
#     print('function')
#     imageDict = {'0':[]}
#     for filename in os.listdir(folder):
#         label = filename[:-4].split('_')[0]
#         img = cv2.imread(os.path.join(folder,filename))
#         if label in imageDict:
#             imageDict[label].append(img)
#         else:
#             imageDict[label] = []
#             imageDict[label].append(img)
#     # for key in sorted(imageDict):
#     #     print(key, len(imageDict[key]))
    
#     newdir = 'p1_' + type
#     if not os.path.exists(newdir):
#         os.makedirs(newdir)

#     for key in imageDict:
#         className = 'class_' + key
#         if not os.path.exists(os.path.join(newdir,className)):
#             os.makedirs(os.path.join(newdir,className))
#         for index, im  in  enumerate(imageDict[key]):
#             cv2.imwrite(os.path.join(newdir,className,key+'_'+str(index))+'.png', im)

#     return 0
class myDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_lists = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):

        fileName = self.img_lists[idx]
        img_path = os.path.join(self.img_dir, fileName)
        image = Image.open(img_path)
        label = fileName[:-4].split('_')[0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = (self.target_transform(label))
        return  (image, int(label))

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    
    model = None
    if model_name == 'vgg13':
        model = torchvision.models.vgg13(pretrained=use_pretrained)
    elif model_name == 'vgg13_bn':
        model = torchvision.models.vgg13_bn(pretrained=use_pretrained)
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=use_pretrained)
    elif model_name == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=use_pretrained)
    elif model_name == 'vgg11':
        model = torchvision.models.vgg11(pretrained=use_pretrained)
    elif model_name == 'vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained=use_pretrained)

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    return model
    # model_ft = torchvision.models.vgg13_bn(pretrained=use_pretrained)
    # set_parameter_requires_grad(model_ft, feature_extract)
    # num_ftrs = model_ft.classifier[6].in_features
    # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    # input_size = 224
    # print(model_ft)
    # return model_ft, input_size

##########################################################

print('start')
# CreateImageFolder('hw1_data/p1_data/train_50', 'train')
# CreateImageFolder('hw1_data/p1_data/val_50', 'val')

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 256)')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs to train (default: 100)')

parser.add_argument('--model_name',type=str, default='vgg13_bn')
parser.add_argument('--save_dir',default='save_models')
parser.add_argument('--checkpoint_dir',default='checkpoints')
args = parser.parse_args()
print(args)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_tfm = transforms.Compose([
    transforms.Resize((142,142)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(), # distortion_scale=0.2, p=0.5
    transforms.RandomCrop(128),
    transforms.ColorJitter(), # brightness=0.15, contrast=0.15, saturation=0.15
    transforms.ToTensor(),
    transforms.RandomErasing()

])
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
# train_set = DatasetFolder('p1_train',loader=lambda x: Image.open(x), extensions="png", transform=train_tfm)
# valid_set = DatasetFolder('p1_val',loader=lambda x: Image.open(x), extensions="png", transform=test_tfm)
train_dataset = myDataset(img_dir='hw1_data/p1_data/train_50',transform=train_tfm)
valid_dataset = myDataset(img_dir='hw1_data/p1_data/val_50',transform=test_tfm)
train_loader = DataLoader(train_dataset, args.batch_size,shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, args.batch_size,shuffle=True, num_workers=0)

# batch_size = 128
# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
# valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

model = initialize_model(model_name=args.model_name,num_classes=50,feature_extract=True)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,8)
iters = len(train_loader)

currentBest = -1
fileName = str(args.batch_size) + '_' + str(args.n_epochs) + '_' + str(args.model_name) + '_' + 'cosineAnneal' + '_' + 'changedata'

for epoch in range(args.n_epochs):
    
    model.train()
    train_loss = []
    train_accs = []
    
    i = 0
    for batch in (train_loader):
        
        imgs, labels = batch
        
        logits = model(imgs.to(device))
        # print(labels)
        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()

        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        optimizer.step()
        
        
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
        train_loss.append(loss.item())
        train_accs.append(acc)
    
        i += 1
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #         }, os.path.join(args.chekpoint_dir,fileName,'.ckpt'))
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):

        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs.to(device))

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    if valid_acc > currentBest:
      
        torch.save(model.state_dict(), os.path.join(args.save_dir,fileName+'.pt') )
        currentBest = valid_acc

    scheduler.step()

print('Best: ', currentBest)

#Best:  tensor(0.7385, device='cuda:0') n_epoch100

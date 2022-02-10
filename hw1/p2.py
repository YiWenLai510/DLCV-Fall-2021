import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import  DataLoader, 
import os
import argparse
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import imageio
import torchvision.transforms.functional as TF


def mean_iou_score(pred, labels, mode):
    '''
    Compute mean IoU score over 6 classes
    '''
    # print(pred.shape,labels.shape)
    pred = torch.argmax(pred,dim=1).squeeze()
    labels = labels.squeeze()
    # print(pred.shape,labels.shape)
    # print(pred.shape)
    # print(labels.shape)
    mean_iou = 0
    # class_iou = []
    for i in range(6):
        tp_fp = torch.sum(pred == i).item()
        tp_fn = torch.sum(labels == i).item()
        tp = torch.sum((pred == i) * (labels == i)).item()
        iou = tp / max(1,tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        if mode == 'val':
            print('class #%d : %1.5f'%(i, iou))
        # class_iou.append(iou)
    if mode == 'val':
        print('\nmean_iou: %f\n' % mean_iou)
    # print('\nloss: %f\n' % loss)
    return mean_iou#, class_iou

def read_masks(file_list,file_dir):
    '''
    Read masks from directory and tranform to categorical
    '''
    # file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    # file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))
    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(file_dir, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 
    return masks

class myDataset(Dataset):
    def __init__(self, img_dir):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        # self.transform = transform
        # self.target_transform = target_transform
        self.img_lists =  [im for im in os.listdir(self.img_dir) if im[:-4].split('_')[1]=='sat' ] 
        self.label_lists =  [im[:-4].split('_')[0] + '_mask.png'  for im in self.img_lists] 
        self.mask_lists = read_masks(self.label_lists, img_dir)  
        # print(self.mask_lists.shape)
        # print(self.img_lists[0],self.label_lists[0])
    def __len__(self):
        return len(self.img_lists)
    def transform(self, image):
        # # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # mask = resize(mask)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # # Random horizontal flipping
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)
        return image #, mask

    def __getitem__(self, idx):

        fileName = self.img_lists[idx]
        # labelName = self.label_lists[idx]
        # labelName = fileName[:-4].split('_')[0] + '_mask.png'
        img_path = os.path.join(self.img_dir, fileName)
        # label_path = os.path.join(self.img_dir, labelName)
        image = Image.open(img_path)
        # label = Image.open(label_path)
        label = self.mask_lists[idx]
        # if self.transform:
        # image = self.transform(image)
        # if self.target_transform:
        # label = self.transform(label)
        # print(label.shape)
        # input()
        return  TF.to_tensor(image), TF.to_tensor(label)



parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training (default: 256)')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='learning rate')
parser.add_argument('--save_dir',default='save_models_p2')
parser.add_argument('--checkpoint_dir',default='checkpoints_p2')
args = parser.parse_args()
print(args)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = torchvision.models.segmentation.fcn_resnet50(num_classes=7)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total Params',pytorch_total_params) #134489934

train_tfm = transforms.ToTensor()

test_tfm = transforms.ToTensor()

train_dataset = myDataset(img_dir='hw1_data/p2_data/train')#train_tfm)
valid_dataset = myDataset(img_dir='hw1_data/p2_data/validation')#test_tfm)
train_loader = DataLoader(train_dataset, args.batch_size,shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, 1, shuffle=True, num_workers=8)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,8)

fileName = '_'.join(('p2', str(args.batch_size), str(args.n_epochs), str(args.lr), 'fcn_resnet50','cosAnneal','GradAccum')  )   
currentBest = -1
accumulation_steps = 4


for epoch in range(args.n_epochs):

    model.train()
    train_loss = []
    train_accs = []
    # i = 0
    for idx, batch in enumerate((train_loader)):
        
        imgs, labels = batch

        labels = torch.squeeze(labels)

        labels = labels.to(device=device,dtype=torch.long)

        logits = model(imgs.to(device))['out']

        loss = criterion(logits, labels) / accumulation_steps
        # optimizer.zero_grad()
        loss.backward()
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        if (idx+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()  
            optimizer.zero_grad()                          # Now we can do an optimizer step
            # model.zero_grad()
        # optimizer.step()  
        if (idx) % accumulation_steps == 0 and idx != 0:      
            acc = mean_iou_score(logits,labels, 'train')
            train_loss.append(loss.item())
            train_accs.append(acc)
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
   
    all_val = []#np.array([])
    all_val_label = []# np.array([])

    for batch in (valid_loader):

        imgs, labels = batch
        # labels = torch.squeeze(labels)
        # print('after squeeze',labels.shape)
        labels = labels.to(device=device,dtype=torch.long)

        with torch.no_grad():
            logits = model(imgs.to(device))['out']
        # logits = torch.squeeze(logits)
        # print('after squeeze',logits.shape, labels.shape)
        all_val.append(logits.cpu())
        all_val_label.append(labels.cpu())

    all_val = torch.cat((all_val),0)
    all_val_label = torch.cat((all_val_label),0)
    
    acc = mean_iou_score(all_val,all_val_label, 'val')

    valid_accs.append(acc)

    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] , acc = {valid_acc:.5f}")

    if valid_acc > currentBest:
      
        torch.save(model.state_dict(), os.path.join(args.save_dir,fileName+'.pt') )
        currentBest = valid_acc
    if epoch%5 == 0:
        torch.save(model.state_dict(), os.path.join(args.save_dir,fileName+'_'+str(epoch)+'.pt'))

    scheduler.step()

print('Best: ', currentBest)

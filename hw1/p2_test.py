import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import PIL
from torch.utils.data import  DataLoader
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
    for i in range(6):
        tp_fp = torch.sum(pred == i).item()
        tp_fn = torch.sum(labels == i).item()
        tp = torch.sum((pred == i) * (labels == i)).item()
        iou = tp / max(1,tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        if mode == 'val':
            print('class #%d : %1.5f'%(i, iou))
    if mode == 'val':
        print('\nmean_iou: %f\n' % mean_iou)
    return mean_iou

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

        self.img_dir = img_dir
        
        self.img_lists =  [im for im in os.listdir(self.img_dir)  ] 
        
    def __len__(self):
        return len(self.img_lists)
    def transform(self, image):
        
        image = TF.to_tensor(image)
        
        return image 
    def __getitem__(self, idx):

        fileName = self.img_lists[idx]
       
        img_path = os.path.join(self.img_dir, fileName)
        
        image = Image.open(img_path)
        
        return  fileName[:-4], TF.to_tensor(image) 

def write_mask(out_dir,logits,fileName):
    

    logits = torch.argmax(logits,dim=1) 

    seg = logits.squeeze()
    # print(seg.shape, seg.dtype)
    mask = torch.Tensor(3,512,512).type(dtype=torch.int64).to(device)

    mask_f = seg.clone()
    mask_f[seg == 0] = 3
    mask_f[seg == 1] = 6
    mask_f[seg == 2] = 5
    mask_f[seg == 3] = 2
    mask_f[seg == 4] = 1
    mask_f[seg == 5] = 7
    mask_f[seg == 6] = 0

    mask[0] = mask_f & 4
    mask[1] = mask_f & 2
    mask[2] = mask_f & 1

    torchvision.utils.save_image(mask.float(), os.path.join(out_dir,fileName)+'.png')
    
    return 0

parser = argparse.ArgumentParser()

parser.add_argument('--test_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()
# print(args)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model_path = 'hw1_2_best.pt'
# class #0 : 0.76701
# class #1 : 0.88572
# class #2 : 0.34077
# class #3 : 0.79784
# class #4 : 0.78810
# class #5 : 0.67226

# mean_iou: 0.708619

model = torchvision.models.segmentation.fcn_resnet50(num_classes=7)
model.load_state_dict(torch.load(args.model_path))

valid_dataset = myDataset(img_dir=args.test_dir)
valid_loader = DataLoader(valid_dataset, 1, shuffle=False, num_workers=8)

model = model.to(device)

model.eval()


for batch in (valid_loader):


    fileName, imgs = batch
    # print(fileName)
    with torch.no_grad():
        logits = model(imgs.to(device))['out']
    # print('after foward')
    write_mask(args.out_dir,logits,fileName[0])





# plot_imgs = [ "hw1_data/p2_data/validation/0010_sat.jpg"
# , "hw1_data/p2_data/validation/0097_sat.jpg"
# , "hw1_data/p2_data/validation/0107_sat.jpg"
# ]
# model_list = ['save_models_p2/p2_4_100_0.0003_fcn_resnet50_cosAnneal_GradAccum_10.pt'
# ,'save_models_p2/p2_4_100_0.0003_fcn_resnet50_cosAnneal_GradAccum_55.pt',
# 'save_models_p2/p2_4_100_0.0003_fcn_resnet50_cosAnneal_GradAccum_85.pt'
# ]
# cc = 0
# for imgpath in plot_imgs:
#     for model_path in model_list:
#         img = Image.open(imgpath)
#         # print(img.size) (512, 512)
#         img = TF.to_tensor(img)
#         # print(img.shape)torch.Size([3, 512, 512])
#         img = torch.unsqueeze(img,0)
#         # print(img.shape) torch.Size([1, 3, 512, 512])   
#         with torch.no_grad():
#             logits = model(img.to(device))['out']

#         logits = torch.argmax(logits,dim=1) 
#         # print(logits.shape) #torch.Size([1, 512, 512])    
       
        
#         img = Image.open(imgpath)   
       
#         seg = logits.cpu()
#         # print(seg.shape, seg.dtype)
#         tmp = torch.Tensor(2,512,512).type(dtype=torch.int64)
#         # print(tmp.shape, tmp.dtype)
#         mask = torch.cat((tmp, seg),0)      
#         # print(mask.shape)
#         seg = seg.squeeze()
#         # print(seg.shape)
        
#         clrMap = {6:'000', 5:'111', 4:'001', 3:'010', 2:'101', 1:'110', 0:'011' }

#         h,w = 512, 512
#         for i in range(h):
#             for j in range(w):
#                 # print(seg[i][j])
#                 to_bin = clrMap[seg[i][j].item()]
#                 # print(i,j,to_bin)
#                 mask[0][i][j] = int(to_bin[0]) 
#                 mask[1][i][j] = int(to_bin[1]) 
#                 mask[2][i][j] = int(to_bin[2])  

#         mask = mask*255
#         tensor = np.array(mask, dtype=np.uint8)
#         # print(tensor.shape)
#         tensor = np.transpose(tensor)
#         # print(tensor)

#         im = PIL.Image.fromarray(tensor)
#         imageio.imsave(str(cc) + '.png', np.uint8(im))
#         cc += 1

import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import json
from tqdm import tqdm
from math import log, e
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  type=str, required=True)
args = parser.parse_args()
model_path = args.model_path

'''dataset'''
class SkullDataset(Dataset):
    def __init__(self, tfm, mode="train") -> None:
        super().__init__()
        self.patient_id = []
        self.data = {}
        self.tfm = tfm
        # record number of different labels & type of patients        
        self.labels = {0:0,1:0,2:0}
        self.patients = {0:0, 1:0}
        self.mode = mode
        self.mx_cor = 0
        if mode == "train":
            with open("./skull/records_train.json", 'r') as json_file:
                datainfo = json.load(json_file)["datainfo"]
                for _, value in datainfo.items():
                    patient_id, img_name = os.path.split(value["path"])
                    img_n = int(img_name[:-4].split("_")[3])
                    img_path = os.path.join('./skull/train/', value["path"])
                    coords = []
                    for coor in value["coords"]:
                        coords.append(img_n)
                        coords.extend(coor)
                        self.mx_cor = max(self.mx_cor, max(coor))
                    label = value["label"] + 1
                    self.labels[label]+=1
                    if patient_id in self.patient_id:
                        self.data[patient_id]["img_path"].append(img_path)
                        self.data[patient_id]["coords"].extend(coords)
                        self.data[patient_id]["label"].append(label)
                    else:
                        self.patients[abs(label-1)]+=1
                        self.patient_id.append(patient_id)
                        self.data[patient_id] = {
                            "img_path": [img_path],
                            "coords": coords,
                            "label": [label]
                            }
        else:
            self.patient_id = sorted(os.listdir("./skull/test/"))
            for patient_id in self.patient_id:
                dir = os.path.join("./skull/test/", patient_id)
                id = sorted(os.listdir(dir))
                self.data[patient_id] = {
                    "id": id,
                    "img_path": [os.path.join(dir, name) for name in id]
                    }
        

    def __len__(self):
        return(len(self.patient_id))

    def __getitem__(self, idx):
        '''
        return image sequence tensor, coordinate sequence, label sequence
        '''
        patient_id = self.patient_id[idx]
        imgs = torch.stack([torch.from_numpy(np.load(path)) for path in self.data[patient_id]["img_path"]]).unsqueeze(1).repeat(1,3,1,1).float()
        imgs += 1024.
        imgs /= 4096.
        if self.mode == "train":
            return self.tfm(imgs), self.data[patient_id]["coords"], torch.LongTensor(self.data[patient_id]["label"])
        else:
            return self.tfm(imgs), self.data[patient_id]["id"]

'''transform'''
train_tfm = transforms.Compose([
    # transforms.RandomResizedCrop((384,384), scale=(0.5, 1.0)),
    # transforms.RandomCrop((350,350),),
    # transforms.ColorJitter(brightness=(0), contrast=(1.5), saturation=(1.5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomErasing(p=1, scale=(0.02, 0.33)),
#     transforms.RandomAdjustSharpness(5, p=1)
])


test_tfm = transforms.Compose([
    transforms.Resize((512, 512))
])

'''train valid set'''
batch_size = 1
val_ratio = 0.1

dataset = SkullDataset(train_tfm, mode="train")
n_train = int(len(dataset) * (1-val_ratio))
n_val = len(dataset) - n_train

train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
# val_set.dataset.tfm = test_tfm
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


'''model'''
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Flatten(1),
        )
        self.feature_extractor.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, 256)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
#             nn.Dropout(),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        x = x.squeeze(0)
        feats  = self.feature_extractor(x)
        feats += torch.Tensor([i/feats.shape[0] for i in range(feats.shape[0])]).unsqueeze(1).to("cuda")
        feats  = feats.unsqueeze(0)
        hid    = self.transformer_encoder(feats)
        hid    = hid.squeeze(0)
        logits = self.classifier(hid)
        return logits

model = MyModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


'''train'''
weight = torch.Tensor([1/13217, 1/14522, 1/4926])
criterion = nn.CrossEntropyLoss(weight.to(device))

learning_rate = 1e-5
l2_reg = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='max')

n_epochs = 1000

cur_lr = learning_rate
best_acc = 0

def entropy(labels, base=None):

    
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = torch.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = torch.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

for epoch in range(n_epochs):
    # ---------- Training ----------
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_acc  = []

    # Iterate the training set by batches.
    train_loader.dataset.dataset.tfm = train_tfm
    for i, batch in enumerate(tqdm(train_loader)):
        
        # forward
        # shape(imgs):   (B, S, C, H, W)
        # shape(feats):  (S, E)
        # shape(hid):    (B, S, E)
        # shape(logits): (S, #class)
        # shape(labels): (S)
        imgs, coords, labels = batch
        logits = model(imgs.to(device))
        ent  = entropy(torch.abs((logits.argmax(dim=-1)-1)))
        loss = criterion(logits, labels.squeeze(0).to(device)) + ent

        # backward & update
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        if (i+1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # record acc & loss
        acc = (logits.argmax(dim=-1) == labels.squeeze(0).to(device)).float().cpu().numpy()
        train_loss.append(loss.item())
        train_acc.append(acc)
        del imgs, coords, labels, logits, loss
        torch.cuda.empty_cache()
    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
#     train_acc = sum(train_acc) / len(train_acc)
    train_acc = np.hstack(train_acc).mean()

    # ---------- Validation ----------
    
    model.eval()
    
    valid_loss = []
    valid_acc  = []

    # Iterate the validation set by batches.
    val_loader.dataset.dataset.tfm = test_tfm
    for batch in tqdm(val_loader):
        
        # forward
        imgs, coords, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.squeeze(0).to(device))
        acc  = (logits.argmax(dim=-1) == labels.squeeze(0).to(device)).float().cpu().numpy()

        # record acc & loss
        valid_loss.append(loss.item())
        valid_acc.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
#     valid_acc = sum(valid_acc) / len(valid_acc)
    valid_acc = np.hstack(valid_acc).mean()
        
    # Print the information.
#     print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    # Print the information.
    print(f"[ Epoch | {epoch + 1:03d}/{n_epochs:03d} ] train loss = {train_loss:.5f}, train acc = {train_acc:.5f}", end=' | ')
    print(f"val loss = {valid_loss:.5f}, val acc = {valid_acc:.5f}")
    
    # Dynamically modify the learning rate
    scheduler.step(train_acc)
    if optimizer.param_groups[0]['lr'] != cur_lr:
        print('learning rate changes from {} to {}'.format(cur_lr, optimizer.param_groups[0]['lr']))
        cur_lr = optimizer.param_groups[0]['lr']
    if cur_lr < 1e-6:
        print('early stopped due to small learning rate')
        break
    
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('✌㋡✌saving model with acc {:.3f}'.format(best_acc))
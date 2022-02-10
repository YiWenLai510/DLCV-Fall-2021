import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  type=str, required=True)
parser.add_argument("--output_csv_path",  type=str, required=True)
args = parser.parse_args()
model_path = args.model_path
output_csv_path = args.output_csv_path

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

test_tfm = transforms.Compose([
    transforms.Resize((512, 512))
])

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
#         self.feature_extractor = models.alexnet()
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
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(),
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
        feats += torch.Tensor([i/feats.shape[0] for i in range(feats.shape[0])]).unsqueeze(1).cuda()
        feats  = feats.unsqueeze(0)
        hid    = self.transformer_encoder(feats)
        hid    = hid.squeeze(0)
        logits = self.classifier(hid)
        return logits

testset = SkullDataset(tfm=test_tfm, mode="test")
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model = MyModel().cuda()
model.load_state_dict(torch.load(model_path))

with open(output_csv_path, "w") as f:
    f.write("id,label,coords\n")
    model.eval()
    for data, ID in testloader:
        data = data.cuda()
        with torch.no_grad():
            logit = model(data)
        label = (logit.argmax(dim=-1)-1).cpu().numpy()
        

        label_abs = np.array([abs(l) for l in label])
        if max(label_abs):
            label[label==0] = -1

        for i, l in zip(ID, label):
            if l == 1:
                f.write(f"{i[0][:-4]},{l},256 256\n")
            else:
                f.write(f"{i[0][:-4]},{l},\n")
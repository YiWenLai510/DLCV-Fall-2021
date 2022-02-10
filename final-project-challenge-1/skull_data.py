import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.utils.rnn as rnn_utils

class SkullDataset(Dataset):
    def __init__(self, mode="train") -> None:
        super().__init__()
        self.mode = mode
        self.patient_id = []
        self.label = {}
        if self.mode == "train":
            with open("./skull/records_train.json", 'r') as json_file:
                data = json.load(json_file)["datainfo"]
                for _, value in data.items():
                    patient_id, img_name = os.path.split(value["path"])
                    p = os.path.join('./skull/train/', value["path"])
                    c = value["coords"]
                    l = max(0, value["label"])
                    if patient_id in self.patient_id:
                        self.label[patient_id]["data"].append(p)
                        self.label[patient_id]["coords"].append(c)
                        self.label[patient_id]["label"].append(l)    # 1: fracture, 0: healthy
                    else:
                        self.patient_id.append(patient_id)
                        self.label[patient_id] = {
                            "data": [p],
                            "coords": [c],
                            "label": [l]
                            }
        else:
            self.patient_id = sorted(os.listdir("./skull/test/"))
            for patient_id in self.patient_id:
                dir = os.path.join("./skull/test/", patient_id)
                id = sorted(os.listdir(dir))
                self.label[patient_id] = {
                    "id": id,
                    "data": [os.path.join(dir, name) for name in id]
                    }

    def __len__(self):
        return(len(self.patient_id))

    def __getitem__(self, idx):
        '''
        return image sequence tensor, coordinate sequence, label sequence
        '''
        patient_id = self.patient_id[idx]
        img = torch.stack([torch.from_numpy(np.load(path)) for path in self.label[patient_id]["data"]]).unsqueeze(1).float()
        if self.mode == "train":
            return img, self.label[patient_id]["coords"], torch.LongTensor(self.label[patient_id]["label"])
        else:
            return img, self.label[patient_id]["id"]

if __name__ == "__main__":
    pass
    
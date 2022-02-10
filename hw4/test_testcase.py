import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        # return image, label
        return image

    def __len__(self):
        return len(self.data_df)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)
class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )
    def forward(self,x):
        x = self.encoder(x)
        return x.view(x.size(0),-1)
    


def euclidean_metric(a, b):
    # print(a.shape)
    # print(b.shape)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    # print(a.shape)
    # print(b.shape)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_similarity(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    # logits = -((a - b)**2).sum(dim=2)
    logits = F.cosine_similarity(a, b, dim=2)
    # print(logits.shape)
    return logits

class Parametric_func(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(3200, 1600),
            nn.ReLU(),
            nn.Linear(1600, 100),
            nn.ReLU(),
            nn.Linear(100,1)
        )
         
        
    def forward(self, a, b): # a: torch.Size([150, 1600]), b: torch.Size([10, 1600])
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1) # torch.Size([150, 10, 1600])
        b = b.unsqueeze(0).expand(n, m, -1)
        c = torch.cat((a,b),2)
        # print(c.shape)
        ans = self.enc(c)
        # print(ans.shape)
        return ans.squeeze()
def predict(args, model, data_loader):

    prediction_results = []
    dis_func = Parametric_func().to(device)
    dis_func.load_state_dict(torch.load('hw4_p1_distance.pt'))

    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        # for i, (data, target) in enumerate(data_loader):
        for i, (data) in enumerate(data_loader):
            # print(i)
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]
            # print(support_input.shape) # torch.Size([5, 3, 84, 84])
            # print(query_input.shape) # torch.Size([75, 3, 84, 84])

            # create the relative label (0 ~ N_way-1) for query data
            # label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            # query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])
            # print(label_encoder)
            # print(query_label)

            # TODO: extract the feature of support and query data
            proto = model(support_input.to(device))
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0)
            # print(proto.shape)
            # TODO: calculate the prototype for each class according to its support data
            # logits = euclidean_metric(model(query_input.to(device)), proto)
            # logits = cosine_similarity(model(query_input.to(device)), proto)
            logits =  dis_func(model(query_input.to(device)), proto)
            # TODO: classify the query data depending on the its distense with each prototype
            pred = torch.argmax(logits, dim=1)
            # print(pred.shape)
            prediction_results.append(pred.detach().cpu())
    
    prediction_results = torch.cat(prediction_results).view(600,-1)
    # print(prediction_results.shape)
    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")\

    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')

    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, default='hw4_data/mini/val.csv', help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, default='hw4_data/mini/val', help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, default='hw4_data/mini/val_testcase.csv', help="Test case csv")
    parser.add_argument('--output_csv', type=str, default='p1_1.csv', help="Output filename")
    # parser.add_argument('--model_path', type=str, default='p1_1.csv', help="Output filename")
    return parser.parse_args()
    # bash hw4_p1.sh hw4_data/mini/val.csv hw4_data/mini/val hw4_data/mini/val_testcase.csv p1_final.csv
if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    model = Convnet().to(device) 
    model.load_state_dict(torch.load(f'hw4_p1_model.pt'))
    prediction_results = predict(args, model, test_loader)

    # TODO: output your prediction to csv
    # h = 'episode_id,query0,query1,query2,query3,query4,query5,query6,query7,query8,query9,query10,query11,query12,query13,query14,query15,query16,query17,query18,query19,query20,query21,query22,query23,query24,query25,query26,query27,query28,query29,query30,query31,query32,query33,query34,query35,query36,query37,query38,query39,query40,query41,query42,query43,query44,query45,query46,query47,query48,query49,query50,query51,query52,query53,query54,query55,query56,query57,query58,query59,query60,query61,query62,query63,query64,query65,query66,query67,query68,query69,query70,query71,query72,query73,query74'
    # 
    testcase = pd.read_csv(args.testcase_csv)
    qs = list(testcase.columns.values)[6:]
    df = pd.DataFrame(prediction_results.numpy(),columns=qs)
    # print(df)
    df.to_csv(args.output_csv)
    # python3 eval.py p1.csv hw4_data/mini/val_testcase_gt.csv


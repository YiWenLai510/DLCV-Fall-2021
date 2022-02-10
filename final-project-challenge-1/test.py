import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import ResLSTM
from skull_data import SkullDataset
from tqdm import tqdm

if __name__ == "__main__":
    testset = SkullDataset(mode="test")
    testloader = DataLoader(testset, batch_size=1, shuffle=False)    # batch_size must be 1 !!!!!!!!!!!!! 因為我沒去把每個seq補成一樣長，就算補成一樣長我GPU memory也不夠吃batch_size > 2
    model = ResLSTM().cuda()
    model.load_state_dict(torch.load("./weights/res34lstm.pt"))

    with open("output.csv", "w") as f:
        f.write("id,label,coords\n")
        model.eval()
        for data, id in testloader:
            data = data.cuda()
            with torch.no_grad():
                logit = model(data)
            label = logit.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()
            if max(label):    # 1
                label = [l if l else -1 for l in label]
            for i, l in zip(id, label):    # no coordinate yet
                f.write(f"{i[0][:-4]},{l},{'0 0' if l == 1 else ''}\n")
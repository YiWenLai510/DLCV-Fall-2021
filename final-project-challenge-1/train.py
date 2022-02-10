import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import ResLSTM
from skull_data import SkullDataset
from tqdm import tqdm

if __name__ == "__main__":
    dataset = SkullDataset()
    n_train = int(len(dataset) * 0.95)
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)    # batch_size must be 1 !!!!!!!!!!!!! 因為我沒去把每個seq補成一樣長，就算補成一樣長我GPU memory也不夠吃batch_size > 2
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    model = ResLSTM( ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 500

    best_acc = 0.83

    for epoch in range(n_epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for data, coord, label in tqdm(train_loader):
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            logit = model(data)
            loss = criterion(logit.squeeze(0), label.squeeze(0))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (logit.argmax(dim=-1) == label).float().mean()

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss / len(train_loader):.5f}, acc = {train_acc / n_train:.5f}")

        val_acc = 0
        model.eval()
        for data, coord, label in tqdm(val_loader):
            data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                logit = model(data)
            val_acc += (logit.argmax(dim=-1) == label).float().mean()

        val_acc /= n_val
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] acc = {val_acc:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            print('Saving model with case-level acc  {:.3f}'.format(best_acc))
            torch.save(model.state_dict(), "./weights/res34lstm.pt")
        
        scheduler.step()
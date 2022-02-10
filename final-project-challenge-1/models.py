import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.utils.rnn as rnn_utils

class ResLSTM(nn.Module):
    def __init__(self):
        super(ResLSTM, self).__init__()
        self.backbone = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.lstm = nn.LSTM(input_size=512, hidden_size=1000, num_layers=3, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        '''
        參考自 https://discuss.pytorch.org/t/how-to-input-image-sequences-to-a-cnn-lstm/89149
        '''
        batch_size, seq_len, C, H, W = x.shape
        feature = self.backbone(x.view(batch_size * seq_len, C, H, W))
        output, hidden = self.lstm(feature.view(batch_size, seq_len , -1))
        pred = self.classifier(output.view(batch_size * seq_len, -1))
        return pred.view(batch_size, seq_len, -1)

if __name__ == "__main__":
    r = torch.randint(-1024, 3071, (2, 1, 512, 512))
    m = ResLSTM()
    print(m)
    o = m(r.float())
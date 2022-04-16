import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 16*4*4)

class LstmModule(nn.Module):
    def __init__(self):
        super(LstmModule, self).__init__()
        self.lstm = nn.LSTM(input_size=16*4*4, hidden_size=96, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(96, 48)
        self.fc2 = nn.Linear(48, 8)
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        hidden, _ = self.lstm(x)
        score = self.fc1(hidden[-1, :, :])
        score = self.fc2(score)
        score = self.sf(score)
        return score

class CnnLstm(BasicModel):
    def __init__(self, args):
        super(CnnLstm, self).__init__(args)
        self.conv = ConvModule()
        self.lstm = LstmModule()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x, embedding, indicator):
        features = self.conv(x.view(-1, 1, 80, 80))
        final_features = features
        score = self.lstm(final_features)
        return score, None
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.modules import dropout
class MyModel(nn.Module):
    def __init__(self, input_size=51, hidden_size=256, num_classes=2):
        super(MyModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=2,dropout=0.4)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pack padded sequences
        out, (ht, ct) = self.lstm1(x)
        # take last time-step output
        out = out[-1,:, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))
        return out
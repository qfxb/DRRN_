import torch
from torch import nn


class DRRN(nn.Module):
    def __init__(self):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x,x2):
        # x = torch.cat((x,x2),dim=1)
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(25):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)
            ...
        out = torch.add(out, residual)
        # out = up(out)
        out = self.output(self.relu(out))

        return out
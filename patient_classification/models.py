import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, cls_num):
        super(MLP, self).__init__()
        self.cls_num = cls_num
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            # nn.Linear(19, 25),
            # nn.ReLU(),
            # nn.Linear(50, 50),
            # nn.ReLU(),
            nn.Linear(self.in_channels, cls_num),
            nn.Softmax()
        )

    def forward(self, x):
        output = self.mlp(x)
        return output

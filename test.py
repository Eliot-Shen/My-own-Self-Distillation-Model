import torch
from torch import nn
from torchinfo import summary
in_channel = 64
mid_channel = 32

net = nn.Sequential(
        nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0),
        nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(mid_channel, in_channel, kernel_size=1, stride=1, padding=0)
    )


def initial(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.1)

x = torch.rand(1,64,56,56)
net.apply(initial)
if __name__=='__main__':
    print(net)
    print(net(x).shape)

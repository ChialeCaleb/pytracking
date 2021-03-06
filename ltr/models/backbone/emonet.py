import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import os


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=1):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x


class Group_Attention(nn.Module):
    def __init__(self):
        super(Group_Attention, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=2, padding=0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=32 * 3, out_channels=64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))

        self.group1 = SpatialGroupEnhance()
        self.group2 = SpatialGroupEnhance()

    def forward(self, x):
        ir1 = self.conv1_1(x)
        ir2 = self.conv1_2(x)
        ir3 = self.conv1_3(x)
        ir = torch.cat([ir1, ir2, ir3], dim=1)

        out1 = self.conv1(ir)
        out1 = self.conv2(out1)
        out1 = self.group1(out1)

        out2 = self.conv3(out1)
        out2 = self.group2(out2)

        return out1, out2


class EventMotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ga = Group_Attention()  # share params

    def forward(self, x1, x2, x3):
        x1_1, x1_2 = self.ga(x1)
        x2_1, x2_2 = self.ga(x2)
        x3_1, x3_2 = self.ga(x3)
        event_layer2=torch.stack([x1_1,x2_1,x3_1],dim=1)
        event_layer3=torch.stack([x1_2,x2_2,x3_2],dim=1)
        return event_layer2,event_layer3


if __name__ == '__main__':
    net = EventMotionNet()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    net = net.cuda()

    var1 = torch.FloatTensor(1, 3, 288, 288).cuda()
    var2 = torch.FloatTensor(1, 3, 288, 288).cuda()
    var3 = torch.FloatTensor(1, 3, 288, 288).cuda()

    out1, out2 = net(var1, var2, var3)

    print('*************')
    print(out1.shape, out2.shape)

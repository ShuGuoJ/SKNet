import torch
from torch import nn

class SKModule(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, reduction=16):
        super(SKModule, self).__init__()
        d = max(out_channel//reduction, 32)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, groups=32),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=2, groups=32, dilation=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.transform = nn.Sequential(
            nn.Linear(out_channel, d),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )
        self.attention_1 = nn.Linear(d, out_channel)
        self.attention_2 = nn.Linear(d, out_channel)

    def forward(self, input):
        # input:[b,h,w,c]
        o_1 = self.branch_1(input)
        o_2 = self.branch_2(input)
        o = o_1 + o_2
        o = self.avgpool(o)
        # o:[b, c]
        o = o.view(o.shape[0], -1)
        o = self.transform(o)
        a_1 = self.attention_1(o)
        a_2 = self.attention_2(o)
        a_1, a_2 = a_1.unsqueeze(-1), a_2.unsqueeze(-1)
        # a:[b,c,2]
        a = torch.cat([a_1, a_2], dim=-1)
        a = torch.softmax(a, dim=-1)
        a_1, a_2 = a[...,0], a[...,1]
        a_1, a_2 = a_1.view(*a_1.shape, 1, 1), a_2.view(*a_2.shape, 1, 1)
        return o_1 * a_1 + o_2 * a_2

class SKBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(SKBottleneck, self).__init__()
        r_channel = out_channel // 2
        self.conv1 = nn.Conv2d(in_channel, r_channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(r_channel)
        self.conv2 = SKModule(r_channel, r_channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(r_channel)
        self.conv3 = nn.Conv2d(r_channel, out_channel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_channel)) if in_channel!=out_channel else None
    def forward(self, input):
        o1 = self.relu(self.bn1(self.conv1(input)))
        o2 = self.relu(self.bn2(self.conv2(o1)))
        o3 = self.relu(self.bn3(self.conv3(o2)))
        x = self.skip(input) if self.skip else input
        return o3 + x

class SKNet(nn.Module):
    def __init__(self, nc):
        super(SKNet, self).__init__()
        depths = [2,2,2,2]
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self.make_layer(depths[0], 64, 256)
        self.layer2 = self.make_layer(depths[1], 256, 512, stride=2)
        self.layer3 = self.make_layer(depths[2], 512, 1024, stride=2)
        self.layer4 = self.make_layer(depths[3], 1024, 2048, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(2048, nc, bias=False)

    def forward(self, input):
        o1 = self.relu(self.bn1(self.conv1(input)))
        o1 = self.maxpool(o1)
        o2 = self.layer1(o1)
        o3 = self.layer2(o2)
        o4 = self.layer3(o3)
        o5 = self.layer4(o4)
        o5 = self.avgpool(o5)
        o5 = o5.view(o5.shape[0], -1)
        o6 = self.classifier(o5)
        return o6

    def make_layer(self, depth, in_channel, out_channel, stride=1):
        blocks = []
        for i in range(depth):
            if not i:
                blocks.append(SKBottleneck(in_channel, out_channel, stride))
            else:
                blocks.append(SKBottleneck(out_channel, out_channel))
        return nn.Sequential(*blocks)

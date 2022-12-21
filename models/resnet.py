'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True, bn=False):
        super(GatedConv2d, self).__init__()
        self.pattern_conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.forward_conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = bn
        if self.bn:
            self.pattern_bn = nn.BatchNorm2d(planes)
            self.forward_bn = nn.BatchNorm2d(planes)
            for p in self.pattern_bn.parameters():
                p.requires_grad = False

        for p in self.pattern_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            if self.bn:
                patterns = (self.pattern_bn(self.pattern_conv(x)) >= 0).float()
            else:
                patterns = (self.pattern_conv(x) >= 0).float()

        forward = self.forward_conv(x)
        if self.bn:
            forward = self.forward_bn(forward)

        return forward*patterns

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, gated_relu=False):
        super(BasicBlock, self).__init__()
        self.gated_relu = gated_relu

        if not self.gated_relu:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.conv1 = GatedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bn=True)
            self.conv2 = GatedConv2d(planes, planes, kernel_size=3, padding=1, bias=False, bn=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if not self.gated_relu:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
        else:
            out = self.conv2(self.conv1(x))
        out += self.shortcut(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, gated_relu=False):
        super(Bottleneck, self).__init__()
        self.gated_relu = gated_relu

        if not self.gated_relu:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion *
                                   planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        else:
            self.conv1 = GatedConv2d(in_planes, planes, kernel_size=1, bias=False, bn=True)
            self.conv2 = GatedConv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False, bn=True)
            self.conv3 = GatedConv2d(planes, sel.fexplansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if not self.gated_relu:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(self.bn3(self.conv3(out)))
        else:
            out = self.conv3(self.conv2(self.conv1(x)))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, gated_relu=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.gated_relu = gated_relu

        if not gated_relu:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = GatedConv2d(3, 64, kernel_size=3, padding=1, bias=False, bn=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, gated_relu=self.gated_relu))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.gated_relu:
            out = self.conv1(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(gated_relu=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], gated_relu=gated_relu)


def ResNet34(gated_relu=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], gated_relu=gated_relu)


def ResNet50(gated_relu=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], gated_relu=gated_relu)


def ResNet101(gated_relu=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], gated_relu=gated_relu)


def ResNet152(gated_relu=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], gated_relu=gated_relu)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

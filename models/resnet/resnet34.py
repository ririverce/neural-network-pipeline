import torch
import torch.nn.functional as F

from models.resnet.resnet_components import ResidualBlock



class ResNet34(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(ResNet34, self).__init__()
        self.input_channels = input_channels
        if type(num_classes) is list:
            self.num_classes = num_classes
        else:
            self.num_classes = [num_classes]
        self.conv1 = torch.nn.Conv2d(self.input_channels, 64,
                                     kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2_1 = ResidualBlock(64, 64)
        self.block2_2 = ResidualBlock(64, 64)
        self.block2_3 = ResidualBlock(64, 64)
        self.block3_1 = ResidualBlock(64, 128, stride=2)
        self.block3_2 = ResidualBlock(128, 128)
        self.block3_3 = ResidualBlock(128, 128)
        self.block3_4 = ResidualBlock(128, 128)
        self.block4_1 = ResidualBlock(128, 256, stride=2)
        self.block4_2 = ResidualBlock(256, 256)
        self.block4_3 = ResidualBlock(256, 256)
        self.block4_4 = ResidualBlock(256, 256)
        self.block4_5 = ResidualBlock(256, 256)
        self.block4_6 = ResidualBlock(256, 256)
        self.block5_1 = ResidualBlock(256, 512, stride=2)
        self.block5_2 = ResidualBlock(512, 512)
        self.block5_3 = ResidualBlock(512, 512)
        self.block5_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(512, sum(self.num_classes))

    def forward(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.pool1(h)
        h = self.block2_1(h)
        h = self.block2_2(h)
        h = self.block2_3(h)
        h = self.block3_1(h)
        h = self.block3_2(h)
        h = self.block3_3(h)
        h = self.block3_4(h)
        h = self.block4_1(h)
        h = self.block4_2(h)
        h = self.block4_3(h)
        h = self.block4_4(h)
        h = self.block4_5(h)
        h = self.block4_6(h)
        h = self.block5_1(h)
        h = self.block5_2(h)
        h = self.block5_3(h)
        h = self.block5_pool(h)
        h = h.view(h.size(0), -1)
        h = self.classifier(h)
        if len(self.num_classes) > 1:
            y = []
            c_start = 0
            for c in self.num_classes:
                y.append(h[:, c_start:c_start+c])
                c_start += c
        else:
            y = h
        return y
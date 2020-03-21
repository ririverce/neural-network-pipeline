import torch
import torch.nn.functional as F

from models.resnet.resnet_components import ResidualBottleneckBlock



class ResNet50(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(ResNet50, self).__init__()
        self.input_channels = input_channels
        if type(num_classes) is list:
            self.num_classes = num_classes
        else:
            self.num_classes = [num_classes]
        self.conv1 = torch.nn.Conv2d(self.input_channels, 64,
                                     kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2_1 = ResidualBottleneckBlock(64, 64, 256)
        self.block2_2 = ResidualBottleneckBlock(256, 64, 256)
        self.block2_3 = ResidualBottleneckBlock(256, 64, 256)
        self.block3_1 = ResidualBottleneckBlock(256, 128, 512, stride=2)
        self.block3_2 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_3 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_4 = ResidualBottleneckBlock(512, 128, 512)
        self.block4_1 = ResidualBottleneckBlock(512, 256, 1024, stride=2)
        self.block4_2 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_3 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_4 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_5 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_6 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block5_1 = ResidualBottleneckBlock(1024, 512, 2048, stride=2)
        self.block5_2 = ResidualBottleneckBlock(2048, 512, 2048)
        self.block5_3 = ResidualBottleneckBlock(2048, 512, 2048)
        self.block5_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(2048, sum(self.num_classes))
 
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
import torch
import torch.nn.functional as F

from models.resnet.resnet_components import ResidualBottleneckBlock



class ResNet151(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(ResNet151, self).__init__()
        self.input_channels = input_channels
        if type(num_classes) is list:
            self.num_classes = num_classes
        else:
            self.num_classes = [num_classes]
        self.conv1 = torch.nn.Conv2d(self.input_channels, 64,
                                     kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2_01 = ResidualBottleneckBlock(64, 64, 256)
        self.block2_02 = ResidualBottleneckBlock(256, 64, 256)
        self.block2_03 = ResidualBottleneckBlock(256, 64, 256)
        self.block3_01 = ResidualBottleneckBlock(256, 128, 512, stride=2)
        self.block3_02 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_03 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_04 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_05 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_06 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_07 = ResidualBottleneckBlock(512, 128, 512)
        self.block3_08 = ResidualBottleneckBlock(512, 128, 512)
        self.block4_01 = ResidualBottleneckBlock(512, 256, 1024, stride=2)
        self.block4_02 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_03 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_04 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_05 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_06 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_07 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_08 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_09 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_10 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_11 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_12 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_13 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_14 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_15 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_16 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_17 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_18 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_19 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_20 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_21 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_22 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_23 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_24 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_25 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_26 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_27 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_28 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_29 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_30 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_31 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_32 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_33 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_34 = ResidualBottleneckBlock(1024, 256, 1024)
        self.block4_35 = ResidualBottleneckBlock(1024, 256, 1024) 
        self.block4_36 = ResidualBottleneckBlock(1024, 256, 1024) 
        self.block5_01 = ResidualBottleneckBlock(1024, 512, 2048, stride=2)
        self.block5_02 = ResidualBottleneckBlock(2048, 512, 2048)
        self.block5_03 = ResidualBottleneckBlock(2048, 512, 2048)
        self.block5_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(2048, sum(self.num_classes))
 
    def forward(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.pool1(h)
        h = self.block2_01(h)
        h = self.block2_02(h)
        h = self.block2_03(h)
        h = self.block3_01(h)
        h = self.block3_02(h)
        h = self.block3_03(h)
        h = self.block3_04(h)
        h = self.block3_05(h)
        h = self.block3_06(h)
        h = self.block3_07(h)
        h = self.block3_08(h)
        h = self.block4_01(h)
        h = self.block4_02(h)
        h = self.block4_03(h)
        h = self.block4_04(h)
        h = self.block4_05(h)
        h = self.block4_06(h)
        h = self.block4_07(h)
        h = self.block4_08(h)
        h = self.block4_09(h)
        h = self.block4_10(h)
        h = self.block4_11(h)
        h = self.block4_12(h)
        h = self.block4_13(h)
        h = self.block4_14(h)
        h = self.block4_15(h)
        h = self.block4_16(h)
        h = self.block4_17(h)
        h = self.block4_18(h)
        h = self.block4_19(h)
        h = self.block4_20(h)
        h = self.block4_21(h)
        h = self.block4_22(h)
        h = self.block4_23(h)
        h = self.block4_24(h)
        h = self.block4_25(h)
        h = self.block4_26(h)
        h = self.block4_27(h)
        h = self.block4_28(h)
        h = self.block4_29(h)
        h = self.block4_30(h)
        h = self.block4_31(h)
        h = self.block4_32(h)
        h = self.block4_33(h)
        h = self.block4_34(h)
        h = self.block4_35(h)
        h = self.block4_36(h)
        h = self.block5_01(h)
        h = self.block5_02(h)
        h = self.block5_03(h)
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
import torch
import torch.nn.functional as F

from models.binary_net.binary_net_components import *



class BinaryNetVGG11(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(BinaryNetVGG11, self).__init__()
        self.input_channels = input_channels
        if type(num_classes) is list:
            self.num_classes = num_classes
        else:
            self.num_classes = [num_classes]
        self.block1_conv1 = BinaryNetConv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.block1_bn1 = torch.nn.BatchNorm2d(64)
        self.block1_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block2_conv1 = BinaryNetConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.block2_bn1 = torch.nn.BatchNorm2d(128)
        self.block2_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block3_conv1 = BinaryNetConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn1 = torch.nn.BatchNorm2d(256)
        self.block3_conv2 = BinaryNetConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn2 = torch.nn.BatchNorm2d(256)
        self.block3_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block4_conv1 = BinaryNetConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn1 = torch.nn.BatchNorm2d(512)
        self.block4_conv2 = BinaryNetConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn2 = torch.nn.BatchNorm2d(512)
        self.block4_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block5_conv1 = BinaryNetConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn1 = torch.nn.BatchNorm2d(512)
        self.block5_conv2 = BinaryNetConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn2 = torch.nn.BatchNorm2d(512)
        self.block5_pool = torch.nn.AdaptiveMaxPool2d((7, 7))
        self.classifier_linear1 = BinaryNetLinear(512*7*7, 4096)
        self.classifier_bn1 = torch.nn.BatchNorm1d(4096)
        self.classifier_linear2 = BinaryNetLinear(4096, 4096)
        self.classifier_bn2 = torch.nn.BatchNorm1d(4096)
        self.classifier_linear3 = BinaryNetLinear(4096, sum(self.num_classes))        

    def forward(self, x):
        h = x
        h = F.relu(self.block1_bn1(self.block1_conv1(h)))
        h = self.block1_pool(h)
        h = F.relu(self.block2_bn1(self.block2_conv1(h)))
        h = self.block2_pool(h)
        h = F.relu(self.block3_bn1(self.block3_conv1(h)))
        h = F.relu(self.block3_bn2(self.block3_conv2(h)))
        h = self.block3_pool(h)
        h = F.relu(self.block4_bn1(self.block4_conv1(h)))
        h = F.relu(self.block4_bn2(self.block4_conv2(h)))
        h = self.block4_pool(h)
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = self.block5_pool(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.classifier_bn1(self.classifier_linear1(h)))
        h = F.relu(self.classifier_bn2(self.classifier_linear2(h)))
        h = self.classifier_linear3(h)
        if len(self.num_classes) > 1:
            y = []
            c_start = 0
            for c in self.num_classes:
                y.append(h[:, c_start:c_start+c])
                c_start += c
        else:
            y = h
        return y
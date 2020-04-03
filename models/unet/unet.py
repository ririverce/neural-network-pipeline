import torch
import torch.nn.functional as F



class UNet(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.block1_conv1 = torch.nn.Conv2d(self.input_channels, 64,
                                            kernel_size=3, stride=1, padding=1)
        self.block1_bn1 = torch.nn.BatchNorm2d(64)
        self.block1_conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block1_bn2 = torch.nn.BatchNorm2d(64)
        self.block1_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block2_conv1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.block2_bn1 = torch.nn.BatchNorm2d(128)
        self.block2_conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.block2_bn2 = torch.nn.BatchNorm2d(128)
        self.block2_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block3_conv1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn1 = torch.nn.BatchNorm2d(256)
        self.block3_conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn2 = torch.nn.BatchNorm2d(256)
        self.block3_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block4_conv1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn1 = torch.nn.BatchNorm2d(512)
        self.block4_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn2 = torch.nn.BatchNorm2d(512)
        self.block4_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block5_conv1 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.block5_bn1 = torch.nn.BatchNorm2d(1024)
        self.block5_conv2 = torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.block5_bn2 = torch.nn.BatchNorm2d(1024)
        self.block5_deconv = torch.nn.ConvTranspose2d(1024, 512,
                                                      kernel_size=2, stride=2, padding=0)
        self.block6_conv1 = torch.nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.block6_bn1 = torch.nn.BatchNorm2d(512)
        self.block6_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block6_bn2 = torch.nn.BatchNorm2d(512)
        self.block6_deconv = torch.nn.ConvTranspose2d(512, 256,
                                                      kernel_size=2, stride=2, padding=0)
        self.block7_conv1 = torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.block7_bn1 = torch.nn.BatchNorm2d(256)
        self.block7_conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block7_bn2 = torch.nn.BatchNorm2d(256)
        self.block7_deconv = torch.nn.ConvTranspose2d(256, 128,
                                                      kernel_size=2, stride=2, padding=0)
        self.block8_conv1 = torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.block8_bn1 = torch.nn.BatchNorm2d(128)
        self.block8_conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.block8_bn2 = torch.nn.BatchNorm2d(128)
        self.block8_deconv = torch.nn.ConvTranspose2d(128, 64,
                                                      kernel_size=2, stride=2, padding=0)
        self.block9_conv1 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.block9_bn1 = torch.nn.BatchNorm2d(64)
        self.block9_conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block9_bn2 = torch.nn.BatchNorm2d(64)
        self.block9_conv3 = torch.nn.Conv2d(64, self.num_classes,
                                            kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        h = x
        h = F.relu(self.block1_bn1(self.block1_conv1(h)))
        h = F.relu(self.block1_bn2(self.block1_conv2(h)))
        f1 = h
        h = self.block1_pool(h)
        h = F.relu(self.block2_bn1(self.block2_conv1(h)))
        h = F.relu(self.block2_bn2(self.block2_conv2(h)))
        f2 = h
        h = self.block2_pool(h)
        h = F.relu(self.block3_bn1(self.block3_conv1(h)))
        h = F.relu(self.block3_bn2(self.block3_conv2(h)))
        f3 = h
        h = self.block3_pool(h)
        h = F.relu(self.block4_bn1(self.block4_conv1(h)))
        h = F.relu(self.block4_bn2(self.block4_conv2(h)))
        f4 = h
        h = self.block4_pool(h)
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = self.block5_deconv(h)
        h = torch.cat([h, f4], 1)
        h = F.relu(self.block6_bn1(self.block6_conv1(h)))
        h = F.relu(self.block6_bn2(self.block6_conv2(h)))
        h = self.block6_deconv(h)
        h = torch.cat([h, f3], 1)
        h = F.relu(self.block7_bn1(self.block7_conv1(h)))
        h = F.relu(self.block7_bn2(self.block7_conv2(h)))
        h = self.block7_deconv(h)
        h = torch.cat([h, f2], 1)
        h = F.relu(self.block8_bn1(self.block8_conv1(h)))
        h = F.relu(self.block8_bn2(self.block8_conv2(h)))
        h = self.block8_deconv(h)
        h = torch.cat([h, f1], 1)
        h = F.relu(self.block9_bn1(self.block9_conv1(h)))
        h = F.relu(self.block9_bn2(self.block9_conv2(h)))
        h = self.block9_conv3(h)
        y = h
        return y
import torch
import torch.nn.functional as F



class FCN8sVGG16(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(FCN8sVGG16, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        """***** VGG16 *****"""
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
        self.block3_conv3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn3 = torch.nn.BatchNorm2d(256)
        self.block3_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block4_conv1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn1 = torch.nn.BatchNorm2d(512)
        self.block4_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn2 = torch.nn.BatchNorm2d(512)
        self.block4_conv3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn3 = torch.nn.BatchNorm2d(512)
        self.block4_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block5_conv1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn1 = torch.nn.BatchNorm2d(512)
        self.block5_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn2 = torch.nn.BatchNorm2d(512)
        self.block5_conv3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn3 = torch.nn.BatchNorm2d(512)
        self.block5_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        """***** additional *****"""
        self.block6_conv1 = torch.nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3)
        self.block6_bn1 = torch.nn.BatchNorm2d(4096)
        self.block7_conv1 = torch.nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
        self.block7_bn1 = torch.nn.BatchNorm2d(4096)
        """***** feature *****"""
        self.up3_conv = torch.nn.Conv2d(256, self.num_classes,
                                        kernel_size=1, stride=1, padding=0)
        self.up3_deconv = torch.nn.ConvTranspose2d(self.num_classes,
                                                   self.num_classes,
                                                   kernel_size=16, stride=8,
                                                   padding=4)
        self.up4_conv = torch.nn.Conv2d(512, self.num_classes,
                                        kernel_size=1, stride=1, padding=0)
        self.up4_deconv = torch.nn.ConvTranspose2d(self.num_classes,
                                                   self.num_classes,
                                                   kernel_size=4, stride=2,
                                                   padding=1)
        self.up7_conv = torch.nn.Conv2d(4096, self.num_classes,
                                        kernel_size=1, stride=1, padding=0)
        self.up7_deconv = torch.nn.ConvTranspose2d(self.num_classes,
                                                   self.num_classes,
                                                   kernel_size=4, stride=2,
                                                   padding=1)

    def forward(self, x):
        h = x
        h = F.relu(self.block1_bn1(self.block1_conv1(h)))
        h = F.relu(self.block1_bn2(self.block1_conv2(h)))
        h = self.block1_pool(h)
        h = F.relu(self.block2_bn1(self.block2_conv1(h)))
        h = F.relu(self.block2_bn2(self.block2_conv2(h)))
        h = self.block2_pool(h)
        h = F.relu(self.block3_bn1(self.block3_conv1(h)))
        h = F.relu(self.block3_bn2(self.block3_conv2(h)))
        h = F.relu(self.block3_bn3(self.block3_conv3(h)))
        h = self.block3_pool(h)
        f3 = h
        h = F.relu(self.block4_bn1(self.block4_conv1(h)))
        h = F.relu(self.block4_bn2(self.block4_conv2(h)))
        h = F.relu(self.block4_bn3(self.block4_conv3(h)))
        h = self.block4_pool(h)
        f4 = h
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = F.relu(self.block5_bn3(self.block5_conv3(h)))
        h = self.block5_pool(h)
        h = F.relu(self.block6_bn1(self.block6_conv1(h)))
        h = F.relu(self.block7_bn1(self.block7_conv1(h)))
        f7 = h
        h = self.up7_conv(f7)
        h = self.up7_deconv(h)
        h = self.up4_conv(f4) + h
        h = self.up4_deconv(h)
        h = self.up3_conv(f3) + h
        h = self.up3_deconv(h)
        y = h
        return y        

        

        

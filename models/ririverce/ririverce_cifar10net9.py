import torch
import torch.nn.functional as F



class RiriverceCifar10Net9(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(RiriverceCifar10Net9, self).__init__()
        self.input_channels = input_channels
        if type(num_classes) is list:
            self.num_classes = num_classes
        else:
            self.num_classes = [num_classes]
        self.conv1 = torch.nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.conv6 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(512)
        self.conv7 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(512)
        self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn8 = torch.nn.BatchNorm2d(512)
        self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.classifier_linear1 = torch.nn.Linear(512, sum(self.num_classes))

    def forward(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.relu(self.bn7(self.conv7(h)))
        h = F.relu(self.bn8(self.conv8(h)))
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        h = self.classifier_linear1(h)
        if len(self.num_classes) > 1:
            y = []
            c_start = 0
            for c in self.num_classes:
                y.append(h[:, c_start:c_start+c])
                c_start += c
        else:
            y = h
        return y
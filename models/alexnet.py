import torch
import torch.nn.functional as F



class AlexNet(torch.nn.Module):

    def __init__(self, num_categories):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.bn_c1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = torch.nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn_c2 = torch.nn.BatchNorm2d(192)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = torch.nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.bn_c3 = torch.nn.BatchNorm2d(384)
        self.conv4 = torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn_c4 = torch.nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_c5 = torch.nn.BatchNorm2d(256)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear1 = torch.nn.Linear(256, 4096)
        self.bn_l1 = torch.nn.BatchNorm1d(4096)
        self.linear2 = torch.nn.Linear(4096, 4096)
        self.bn_l2 = torch.nn.BatchNorm1d(4096)
        self.linear3 = torch.nn.Linear(4096, num_categories)

    def forward(self, x):
        h = x
        h = F.relu(self.bn_c1(self.conv1(h)))
        h = self.pool1(h)
        h = F.relu(self.bn_c2(self.conv2(h)))
        h = self.pool2(h)
        h = F.relu(self.bn_c3(self.conv3(h)))
        h = F.relu(self.bn_c4(self.conv4(h)))
        h = F.relu(self.bn_c5(self.conv5(h)))
        h = self.pool5(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.bn_l1(self.linear1(h)))
        h = F.relu(self.bn_l2(self.linear2(h)))
        h = self.linear3(h)
        y = h
        return y


""" For BengaliAI-CV19 (kaggle) """
class BengaliAlexNet(torch.nn.Module):

    def __init__(self, num_categories):
        super(BengaliAlexNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.bn_c1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = torch.nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn_c2 = torch.nn.BatchNorm2d(192)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = torch.nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.bn_c3 = torch.nn.BatchNorm2d(384)
        self.conv4 = torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn_c4 = torch.nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_c5 = torch.nn.BatchNorm2d(256)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear1 = torch.nn.Linear(256*4*4, 4096)
        self.bn_l1 = torch.nn.BatchNorm1d(4096)
        self.linear2 = torch.nn.Linear(4096, 4096)
        self.bn_l2 = torch.nn.BatchNorm1d(4096)
        self.linear3_graph = torch.nn.Linear(4096, num_categories[0])
        self.linear3_vowel = torch.nn.Linear(4096, num_categories[1])
        self.linear3_conso = torch.nn.Linear(4096, num_categories[2])

    def forward(self, x):
        h = x
        h = F.relu(self.bn_c1(self.conv1(h)))
        h = self.pool1(h)
        h = F.relu(self.bn_c2(self.conv2(h)))
        h = self.pool2(h)
        h = F.relu(self.bn_c3(self.conv3(h)))
        h = F.relu(self.bn_c4(self.conv4(h)))
        h = F.relu(self.bn_c5(self.conv5(h)))
        h = self.pool5(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.bn_l1(self.linear1(h)))
        h = F.relu(self.bn_l2(self.linear2(h)))
        h_graph = F.softmax(self.linear3_graph(h), dim=-1)
        h_vowel = F.softmax(self.linear3_vowel(h), dim=-1)
        h_conso = F.softmax(self.linear3_conso(h), dim=-1)
        y = [h_graph, h_vowel, h_conso]
        return y
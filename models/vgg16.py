import torch
import torch.nn.functional as F



class VGG16(torch.nn.Module):

    def __init__(self, num_categories):
        super(VGG16, self).__init__()
        self.num_categories = num_categories
        self.block1_conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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
        self.classifier_linear1 = torch.nn.Linear(512*7*7, 4096)
        self.classifier_bn1 = torch.nn.BatchNorm1d(4096)
        self.classifier_linear2 = torch.nn.Linear(4096, 4096)
        self.classifier_bn2 = torch.nn.BatchNorm1d(4096)
        self.classifier_linear3 = torch.nn.Linear(4096, num_categories)        

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
        h = F.relu(self.block4_bn1(self.block4_conv1(h)))
        h = F.relu(self.block4_bn2(self.block4_conv2(h)))
        h = F.relu(self.block4_bn3(self.block4_conv3(h)))
        h = self.block4_pool(h)
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = F.relu(self.block5_bn3(self.block5_conv3(h)))
        h = self.block5_pool(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.classifier_bn1(self.classifier_linear1(h)))
        h = F.relu(self.classifier_bn2(self.classifier_linear2(h)))
        h = self.classifier_linear3(h)
        h = F.softmax(h, dim=-1)
        y = h
        return y



class BengaliVGG16(torch.nn.Module):

    def __init__(self, num_categories):
        super(BengaliVGG16, self).__init__()
        self.num_categories = num_categories
        self.block1_conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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
        self.classifier_graph_linear1 = torch.nn.Linear(512*4*4, 4096)
        self.classifier_graph_bn1 = torch.nn.BatchNorm1d(4096)
        self.classifier_graph_linear2 = torch.nn.Linear(4096, 4096)
        self.classifier_graph_bn2 = torch.nn.BatchNorm1d(4096)
        self.classifier_graph_linear3 = torch.nn.Linear(4096, num_categories[0])
        self.classifier_vowel_linear1 = torch.nn.Linear(512*4*4, 4096)
        self.classifier_vowel_bn1 = torch.nn.BatchNorm1d(4096)
        self.classifier_vowel_linear2 = torch.nn.Linear(4096, 4096)
        self.classifier_vowel_bn2 = torch.nn.BatchNorm1d(4096)
        self.classifier_vowel_linear3 = torch.nn.Linear(4096, num_categories[1])
        self.classifier_conso_linear1 = torch.nn.Linear(512*4*4, 4096)
        self.classifier_conso_bn1 = torch.nn.BatchNorm1d(4096)
        self.classifier_conso_linear2 = torch.nn.Linear(4096, 4096)
        self.classifier_conso_bn2 = torch.nn.BatchNorm1d(4096)
        self.classifier_conso_linear3 = torch.nn.Linear(4096, num_categories[2])


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
        h = F.relu(self.block4_bn1(self.block4_conv1(h)))
        h = F.relu(self.block4_bn2(self.block4_conv2(h)))
        h = F.relu(self.block4_bn3(self.block4_conv3(h)))
        h = self.block4_pool(h)
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = F.relu(self.block5_bn3(self.block5_conv3(h)))
        h = self.block5_pool(h)
        h = h.view(h.size(0), -1)
        h_graph = F.relu(self.classifier_graph_bn1(self.classifier_graph_linear1(h)))
        h_graph = F.relu(self.classifier_graph_bn2(self.classifier_graph_linear2(h_graph)))
        h_graph = self.classifier_graph_linear3(h_graph)
        #h_graph = F.softmax(h_graph, dim=-1)
        h_vowel = F.relu(self.classifier_vowel_bn1(self.classifier_vowel_linear1(h)))
        h_vowel = F.relu(self.classifier_vowel_bn2(self.classifier_vowel_linear2(h_vowel)))
        h_vowel = self.classifier_vowel_linear3(h_vowel)
        #h_vowel = F.softmax(h_vowel, dim=-1)
        h_conso = F.relu(self.classifier_conso_bn1(self.classifier_conso_linear1(h)))
        h_conso = F.relu(self.classifier_conso_bn2(self.classifier_conso_linear2(h_conso)))
        h_conso = self.classifier_conso_linear3(h_conso)
        #h_conso = F.softmax(h_conso, dim=-1)
        y = [h_graph, h_vowel, h_conso]
        return y




'''*************
***** TEST *****
*************'''
def unit_test():
    print("[VGG16]")
    import numpy as np
    net = VGG16(1000)
    image = np.zeros((8, 3, 224, 224)).astype(np.float32)
    image = torch.from_numpy(image)
    print("input shape :")
    print(image.shape)
    pred = net(image)
    print("output shape :")
    print(pred.shape)

if __name__ == "__main__":
    unit_test()
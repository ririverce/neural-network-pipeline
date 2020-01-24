import torch
import torch.nn.functional as F



class SSD300(torch.nn.Module):

    def __init__(self, num_categories, num_bboxes):
        super(SSD300, self).__init__()
        self.num_categories = num_categories
        self.num_bboxes = num_bboxes
        '''**************
        ***** VGG16 *****
        **************'''
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
        self.block5_pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        '''**************************
        ***** SSD300 Extensions *****
        **************************'''      
        self.block6_conv1 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, dilation=6, padding=6)
        self.block6_bn1 = torch.nn.BatchNorm2d(1024)
        self.block7_conv1 = torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.block7_bn1 = torch.nn.BatchNorm2d(1024)
        self.block8_conv1 = torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.block8_bn1 = torch.nn.BatchNorm2d(256)
        self.block8_conv2 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.block8_bn2 = torch.nn.BatchNorm2d(512)
        self.block9_conv1 = torch.nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.block9_bn1 = torch.nn.BatchNorm2d(128)
        self.block9_conv2 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.block9_bn2 = torch.nn.BatchNorm2d(256)
        self.block10_conv1 = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.block10_bn1 = torch.nn.BatchNorm2d(128)
        self.block10_conv2 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.block10_bn2 = torch.nn.BatchNorm2d(256)
        self.block11_conv1 = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.block11_bn1 = torch.nn.BatchNorm2d(128)
        self.block11_conv2 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.block11_bn2 = torch.nn.BatchNorm2d(256)
        '''*****************
        ***** multibox *****
        *****************'''
        num_conf = num_bboxes[0] * num_categories
        num_loc = num_bboxes[0] * 4
        self.conf_block4_conv3 = torch.nn.Conv2d(512, num_conf, kernel_size=3, stride=1, padding=1)
        self.loc_block4_conv3 = torch.nn.Conv2d(512, num_loc, kernel_size=3, stride=1, padding=1)
        num_conf = num_bboxes[1] * num_categories
        num_loc = num_bboxes[1] * 4
        self.conf_block7_conv1 = torch.nn.Conv2d(1024, num_conf, kernel_size=3, stride=1, padding=1)
        self.loc_block7_conv1 = torch.nn.Conv2d(1024, num_loc, kernel_size=3, stride=1, padding=1)
        num_conf = num_bboxes[2] * num_categories
        num_loc = num_bboxes[2] * 4
        self.conf_block8_conv2 = torch.nn.Conv2d(512, num_conf, kernel_size=3, stride=1, padding=1)
        self.loc_block8_conv2 = torch.nn.Conv2d(512, num_loc, kernel_size=3, stride=1, padding=1)
        num_conf = num_bboxes[3] * num_categories
        num_loc = num_bboxes[3] * 4
        self.conf_block9_conv2 = torch.nn.Conv2d(256, num_conf, kernel_size=3, stride=1, padding=1)
        self.loc_block9_conv2 = torch.nn.Conv2d(256, num_loc, kernel_size=3, stride=1, padding=1)
        num_conf = num_bboxes[4] * num_categories
        num_loc = num_bboxes[4] * 4
        self.conf_block10_conv2 = torch.nn.Conv2d(256, num_conf, kernel_size=3, stride=1, padding=1)
        self.loc_block10_conv2 = torch.nn.Conv2d(256, num_loc, kernel_size=3, stride=1, padding=1)
        num_conf = num_bboxes[5] * num_categories
        num_loc = num_bboxes[5] * 4
        self.conf_block11_conv2 = torch.nn.Conv2d(256, num_conf, kernel_size=3, stride=1, padding=1)
        self.loc_block11_conv2 = torch.nn.Conv2d(256, num_loc, kernel_size=3, stride=1, padding=1)

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
        f4 = h
        h = self.block4_pool(h)
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = F.relu(self.block5_bn3(self.block5_conv3(h)))
        h = self.block5_pool(h)
        h = F.relu(self.block6_bn1(self.block6_conv1(h)))
        h = F.relu(self.block7_bn1(self.block7_conv1(h)))
        f7 = h
        h = F.relu(self.block8_bn1(self.block8_conv1(h)))
        h = F.relu(self.block8_bn2(self.block8_conv2(h)))
        f8 = h
        h = F.relu(self.block9_bn1(self.block9_conv1(h)))
        h = F.relu(self.block9_bn2(self.block9_conv2(h)))
        f9 = h
        h = F.relu(self.block10_bn1(self.block10_conv1(h)))
        h = F.relu(self.block10_bn2(self.block10_conv2(h)))
        f10 = h
        h = F.relu(self.block11_bn1(self.block11_conv1(h)))
        h = F.relu(self.block11_bn2(self.block11_conv2(h)))
        f11 = h
        conf = [self.conf_block4_conv3(f4), self.conf_block7_conv1(f7),
                self.conf_block8_conv2(f8), self.conf_block9_conv2(f9),
                self.conf_block10_conv2(f10), self.conf_block11_conv2(f11)]
        conf = [l.view(l.size(0), -1, self.num_categories) for l in conf]
        conf = torch.cat(conf, 1)
        loc = [self.loc_block4_conv3(f4), self.loc_block7_conv1(f7),
               self.loc_block8_conv2(f8), self.loc_block9_conv2(f9),
               self.loc_block10_conv2(f10), self.loc_block11_conv2(f11)]
        loc = [l.view(l.size(0), -1, 4) for l in loc]
        loc = torch.cat(loc, 1)
        return conf, loc



'''*************
***** TEST *****
*************'''
def unit_test():
    print("[SSD300]")
    import numpy as np
    net = SSD300(10, (4, 6, 6, 6, 4, 4))
    image = np.zeros((8, 3, 300, 300)).astype(np.float32)
    image = torch.from_numpy(image)
    print("input shape :")
    print(image.shape)
    pred = net(image)
    print("output shape :")
    print(pred[0].shape)
    print(pred[1].shape)

if __name__ == "__main__":
    unit_test()
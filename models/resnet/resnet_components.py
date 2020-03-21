import torch
import torch.nn.functional as F



class ResidualBlock(torch.nn.Module):

    def __init__(self, input_fmaps, output_fmaps, stride=1):
        super(ResidualBlock, self).__init__()
        """ check """
        if output_fmaps % input_fmaps != 0:
            raise ValueError("error!")
        self.input_fmaps = input_fmaps
        self.output_fmaps = output_fmaps
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(self.input_fmaps, self.output_fmaps,
                                     kernel_size=3, stride=self.stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(self.output_fmaps)
        self.conv2 = torch.nn.Conv2d(self.output_fmaps, self.output_fmaps,
                                     kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.output_fmaps)
        if self.input_fmaps != self.output_fmaps:
            self.conv_x = torch.nn.Conv2d(input_fmaps, output_fmaps,
                                          kernel_size=1, stride=self.stride, padding=0)
            self.bn_x = torch.nn.BatchNorm2d(output_fmaps)

    def forward(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.bn2(self.conv2(h))
        x = F.relu(self.bn_x(self.conv_x(x))) if self.input_fmaps != self.output_fmaps else x
        h = h + x
        h = F.relu(h)
        y = h
        return y



class ResidualBottleneckBlock(torch.nn.Module):

    def __init__(self, input_fmaps, mid_fmaps, output_fmaps, stride=1):
        super(ResidualBottleneckBlock, self).__init__()
        """ check """
        if output_fmaps % input_fmaps != 0:
            raise ValueError("error!")
        self.input_fmaps = input_fmaps
        self.mid_fmaps = mid_fmaps
        self.output_fmaps = output_fmaps
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(self.input_fmaps, self.mid_fmaps,
                                     kernel_size=1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(self.mid_fmaps)
        self.conv2 = torch.nn.Conv2d(self.mid_fmaps, self.mid_fmaps,
                                     kernel_size=3, stride=self.stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.mid_fmaps)
        self.conv3 = torch.nn.Conv2d(self.mid_fmaps, self.output_fmaps,
                                     kernel_size=1, stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(self.output_fmaps)
        if self.input_fmaps != self.output_fmaps or self.stride > 2:
            self.conv_x = torch.nn.Conv2d(self.input_fmaps, self.output_fmaps,
                                          kernel_size=1, stride=self.stride, padding=0)
            self.bn_x = torch.nn.BatchNorm2d(self.output_fmaps)

    def forward(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        if self.input_fmaps != self.output_fmaps or self.stride > 1:
            x = F.relu(self.bn_x(self.conv_x(x)))
        h = h + x
        h = F.relu(h)
        y = h
        return y
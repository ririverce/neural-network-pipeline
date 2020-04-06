import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from torch._jit_internal import List



class BinaryNetLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BinaryNetLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = 2 * (self.weight >= 0.0).float() - 1
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class BinaryNetConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BinaryNetConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        weight = 2 * (self.weight >= 0.0).float() - 1
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)



def binary_hard_tanh(input):
    hard_tanh = (input.abs() < 1.0) * input + (input.abs() >= 1.0) * input / input.abs()
    hard_tanh_np = hard_tanh.detach().cpu().numpy()
    diff = (hard_tanh_np > 0.0) * (hard_tanh_np < 1.0) * (1.0 - hard_tanh_np)
    diff -= (hard_tanh_np <= 0.0) * (hard_tanh_np > -1.0) * (hard_tanh_np + 1.0)
    result = hard_tanh + torch.from_numpy(diff).to(hard_tanh.device)
    return result
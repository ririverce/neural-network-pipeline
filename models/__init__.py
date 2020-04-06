"""***********************
***** Classification *****
***********************"""

"""***** VGG *****"""
from models.vgg.vgg11 import VGG11
from models.vgg.vgg13 import VGG13
from models.vgg.vgg16 import VGG16
from models.vgg.vgg19 import VGG19

"""***** ResNet *****"""
from models.resnet.resnet18 import ResNet18
from models.resnet.resnet34 import ResNet34
from models.resnet.resnet50 import ResNet50
from models.resnet.resnet101 import ResNet101
from models.resnet.resnet151 import ResNet151

"""***** Binary Net *****"""
#from models.binary_net.binary_components import BinaryNetLinear
from models.binary_net.binary_net_vgg11 import BinaryNetVGG11



"""**************************************
***** Object Detection (anchor box) *****
**************************************"""
from models.ssd.ssd300_vgg16 import SSD300VGG16
from models.ssd.ssd300_lite_vgg16 import SSD300LiteVGG16



"""******************************
***** Semantic Segmentation *****
******************************"""

"""***** FCN *****"""
from models.fcn.fcn32s_vgg16 import FCN32sVGG16
from models.fcn.fcn16s_vgg16 import FCN16sVGG16
from models.fcn.fcn8s_vgg16 import FCN8sVGG16

"""***** UNet *****"""
from models.unet.unet import UNet
from models.unet.lite_unet import LiteUNet



"""***************
***** Others *****
***************"""
from models.ririverce.ririverce_cifar10net9 import RiriverceCifar10Net9
from models.ririverce.binary_ririverce_cifar10 import BinaryRiriverceCifar10
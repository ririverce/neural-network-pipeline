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



"""**************************************
***** Object Detection (anchor box) *****
**************************************"""
from models.ssd.ssd300_vgg16 import SSD300VGG16
from models.ssd.ssd300_lite_vgg16 import SSD300LiteVGG16



"""***************
***** Others *****
***************"""
from models.ririverce.ririverce_cifar10net9 import RiriverceCifar10Net9
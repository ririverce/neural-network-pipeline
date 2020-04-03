import random
import copy

import numpy as np
import cv2



def random_contrast(image, scale_range=(0.5, 1.5), prob=1.0):
    if random.random() > prob:
        return image
    scale_min, scale_max = scale_range
    scale = random.uniform(scale_min, scale_max)
    image = image * scale
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def random_hue(image, deg_range=(-45, 45), prob=1.0):
    if random.random() > prob:
        return image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    deg = random.uniform(deg_range[0], deg_range[1])
    image[:, :, 0] = np.mod(
                         image[:, :, 0].astype(np.float) + int(deg / 2) + 180,
                         180
                     ).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

"""***************
***** Filter *****
***************"""
def random_blur(image, blur_type="gaussian",
                min_size=1, max_size=11, prob=1.0):
    if random.random() > prob:
        return image
    random_size = random.randrange(min_size, max_size+1, 2)
    if blur_type == "gaussian":
        image = cv2.GaussianBlur(image, (random_size, random_size), 0)
    elif blur_type == "median":
        image = cv2.medianBlur(image, random_size)
    elif blur_type == "bilateral":
        image = cv2.bilateralFilter(image, random_size, 1, 1)
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def add_gaussian_noise(image, max_var=10, prob=1.0):
    if random.random() > prob:
        return image
    var = random.random() * max_var
    gauss = np.random.normal(0, var, image.shape)
    image = image + gauss
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image



"""***************
***** Affine *****
***************"""
def random_rotate(image, deg_range=(-180, 180), prob=1.0):
    if random.random() > prob:
        return image
    height, width, channel = image.shape
    bg_image = np.zeros([height*3, width*3, channel], dtype=np.uint8)
    bg_image[height:height*2, width:width*2] = image
    bg_image[height:height*2, :width] = image[:, ::-1]
    bg_image[height:height*2, width*2:] = image[:, ::-1]  
    bg_image[:height] = bg_image[height:height*2][::-1]  
    bg_image[height*2:] = bg_image[height:height*2][::-1]
    bg_height, bg_width = bg_image.shape[:2]
    bg_center = (int(bg_width/2), int(bg_height/2))
    deg_min, deg_max = deg_range    
    angle = random.uniform(deg_min, deg_max)
    angle += 360 * (angle < 0)    
    matrix = cv2.getRotationMatrix2D(bg_center, angle, 1.0)
    bg_image = cv2.warpAffine(bg_image, matrix, (bg_width, bg_height))
    image = bg_image[height:height*2, width:width*2]
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def random_shift(image, max_shift_ratio=(0.2, 0.2), prob=1.0):
    if random.random() > prob:
        return image
    image = random_horizontal_shift(image,
                                    max_shift_ratio=max_shift_ratio[0],
                                    prob=1.0)
    image = random_vertical_shift(image,
                                  max_shift_ratio=max_shift_ratio[1],
                                  prob=1.0)
    return image

def random_horizontal_shift(image, max_shift_ratio=0.2, prob=1.0):
    if random.random() > prob:
        return image
    height, width, channel = image.shape
    shift = int(random.random() * max_shift_ratio * width)
    bg_image = np.zeros([height, width+shift, channel], dtype=np.uint8)
    if random.random() > 0.5:
        bg_image[:, :width] = image
        bg_image[:, width:] = image[:, width-shift:][::-1]
        image = bg_image[:, shift:]
    else:
        bg_image[:, shift:] = image
        bg_image[:, :shift] = image[:, :shift][::-1]
        image = bg_image[:, :width]
    return image
    
def random_vertical_shift(image, max_shift_ratio=0.2, prob=1.0):
    if random.random() > prob:
        return image
    height, width, channel = image.shape
    shift = int(random.random() * max_shift_ratio * height)
    bg_image = np.zeros([height+shift, width, channel], dtype=np.uint8)
    if random.random() > 0.5:
        bg_image[:height] = image
        bg_image[height:] = image[height-shift:][::-1]
        image = bg_image[shift:]
    else:
        bg_image[shift:] = image
        bg_image[:shift] = image[:shift][::-1]
        image = bg_image[:height]
    return image

def random_distortion(image, shear_range=0.2, prob=1.0):
    """ https://blog.shikoan.com/opencv-distortion/ """
    if random.random() > prob:
        return image
    h = image.shape[0] // 2
    w = image.shape[1] // 2
    randoms = np.random.uniform(1.0 - shear_range,
                                1.0 + shear_range,
                                (3,2)).astype(np.float32)
    coefs = np.array([[-1,-1],[1,-1],[1,1]], np.float32)
    centers = np.array([[h,w]], np.float32)
    origin = centers + centers * coefs
    dest = centers + centers * coefs * randoms
    affine_matrix = cv2.getAffineTransform(origin, dest)
    image = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image



"""***************
***** Cutout *****
***************"""
def random_cutout(image,
                  min_size_ratio=0.1, max_size_ratio=0.1,
                  min_num_cut=1, max_num_cut=5, prob=0.2):
    if random.random() > prob:
        return image
    image_height, image_width = image.shape[:2]
    num_cut = random.randint(min_num_cut, max_num_cut)
    for i in range(num_cut):
        rand = random.uniform(min_size_ratio, max_size_ratio)
        width = image_height * rand
        rand = random.uniform(min_size_ratio, max_size_ratio)
        height = image_height * rand
        crop_x_min = (image_width - width) * random.random()
        crop_y_min = (image_height - height) * random.random()
        crop_x_max = width + crop_x_min
        crop_y_max = height + crop_y_min
        #color = random.random() * 255
        color = 0
        image[int(crop_y_min):int(crop_y_max), int(crop_x_min):int(crop_x_max)] = int(color)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image



"""**************
***** Mixup *****
**************"""
def random_mixup(image_a, image_b, target_a, target_b, alpha=1.0, beta=1.0, prob=1.0):
    if random.random() > prob:
        return image_a, target_a
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)
    coefficient = np.random.beta(alpha, beta)
    coefficient = coefficient if coefficient > 0.5 else 1 - coefficient
    image = coefficient * image_a + (1 - coefficient) * image_b
    target = coefficient * target_a + (1 - coefficient) * target_b
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, target

from torchvision import transforms
import PIL.Image
import torch
import random
import math
import numpy as np


def get_transform(is_eval, net_type, trans):
    
    if trans == 'GLorig' and net_type == 'bn_inception':
        trans = GL_orig_RE_Inception(is_train=not is_eval, RE=False)
    elif trans == 'GL_orig_RE' and net_type == 'bn_inception':
        trans = GL_orig_RE_Inception(is_train=not is_eval, RE=True)
    elif trans == 'GLorig':
        trans = GL_orig_RE(is_train=not is_eval, RE=False)
    elif trans == 'GL_orig_RE':
        trans = GL_orig_RE(is_train=not is_eval, RE=True)
    return trans


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def std_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).std(dim=1)


def mean_per_channel(images):
    images = torch.stack(images, dim=0)
    return images.view(3, -1).mean(dim=1)


class Identity():  # used for skipping transforms
    def __call__(self, im):
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (
                         tensor - self.in_range[0]
                 ) / (
                         self.in_range[1] - self.in_range[0]
                 ) * (
                         self.out_range[1] - self.out_range[0]
                 ) + self.out_range[0]
        return tensor

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


def GL_orig_RE(sz_crop=[384, 128], mean=[0.485, 0.456, 0.406],
                std=[0.299, 0.224, 0.225], is_train=True, RE=False):

    #sz_resize = 288
    #sz_crop = 256
    
    sz_resize = 256
    sz_crop = 227
     
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    if is_train and RE:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5,
                          mean=(0.4914, 0.4822, 0.4465))
        ])
    elif is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(sz_resize),
            transforms.CenterCrop(sz_crop),
            transforms.ToTensor(),
            normalize_transform
        ])
    
    print(transform)

    return transform


def GL_orig_RE_Inception(sz_crop=[384, 128], mean=[104, 117, 128],
                         std=[1, 1, 1], is_train=True, RE=False):

    sz_resize = 256
    sz_crop = 227
    normalize_transform = transforms.Normalize(mean=mean, std=std)

    if is_train and RE:
        transform = transforms.Compose([
            RGBToBGR(),
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]),
            normalize_transform,
            RandomErasing(probability=0.5,
                          mean=(0.4914, 0.4822, 0.4465))
        ])
    elif is_train:
        transform = transforms.Compose([
            RGBToBGR(),
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]),
            normalize_transform
        ])
    else:
        transform = transforms.Compose([
            RGBToBGR(),
            transforms.Resize(sz_resize),
            transforms.CenterCrop(sz_crop),
            transforms.ToTensor(),
            ScaleIntensities([0, 1], [0, 255]),
            normalize_transform
        ])
    
    print(transform)

    return transform


class RandomErasing(object):
    """
    From https://github.com/zhunzhong07/Random-Erasing
    Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al.
    See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img



import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import copy
import logging
import pandas as pd
from .utils import get_transform, pil_loader


logger = logging.getLogger('GNNReID.Dataset')


class Birds_DML(torch.utils.data.Dataset):
    def __init__(self, root, labels, transform=None,
                 eval_reid=False, net_type='resnet50'):
        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        self.eval_reid = eval_reid
        self.trans = transform
        self.net_type = net_type
        self.ys, self.im_paths = [], []
        
        for i in torchvision.datasets.ImageFolder(
                root=os.path.join(root, 'images')
        ).imgs:
            # i[1]: label, i[0]: path to file, including root
            y = i[1]
            fn = os.path.split(i[0])[1]
            
            if y in self.labels and fn[:2] != '._':
                self.ys += [y]
                self.im_paths.append(i[0])
        
        self.transform = get_transform(self.eval_reid, self.net_type, self.trans)

        self.path_to_ind = {p: i for i, p in enumerate(self.im_paths)}

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        im = self.transform(im)
       
        return im, self.ys[index], index, self.im_paths[index]


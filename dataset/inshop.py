import PIL.Image
import torch
import numpy as np
import pandas as pd
import os
from .utils import get_transform, pil_loader


class Inshop_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=None, eval_reid=False, net_type='resnet50'):

        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []

        data_info = np.array(pd.read_table(root +'/Eval/list_eval_partition.txt', sep='\s+', header=1))[:,:]
        #Separate into training dataset and query/gallery dataset for testing.
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        #Generate conversions
        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

        #Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
        for img_path, key in train:
            self.train_im_paths.append(os.path.join(root, 'Img', img_path))
            self.train_ys += [int(key)]

        for img_path, key in query:
            self.query_im_paths.append(os.path.join(root, 'Img', img_path))
            self.query_ys += [int(key)]

        for img_path, key in gallery:
            self.gallery_im_paths.append(os.path.join(root, 'Img', img_path))
            self.gallery_ys += [int(key)]

        if mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys
        elif mode == 'gnn':
            self.im_paths = self.query_im_paths + self.gallery_im_paths
            self.ys = self.query_ys + self.gallery_ys

        self.path_to_ind = {p: i for i, p in enumerate(self.im_paths)}

        self.transform = get_transform(eval_reid, net_type, transform)

    def nb_classes(self):
        return len(set(self.ys))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        im = self.transform(im)

        return im, self.ys[index], index, self.im_paths[index]



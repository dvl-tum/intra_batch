import dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, KReciprocalSampler, KReciprocalSamplerInshop, ClusterSampler
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import copy

logger = logging.getLogger('GNNReID.DataUtility')


def create_loaders(data_root, num_workers, num_classes_iter=None,
                   num_elements_class=None, trans='norm', num_classes=None, 
                   net_type='resnet50', bssampling=None, mode='train'):
    size_batch = num_classes_iter * num_elements_class
    
    dl_tr = get_train_loaders(data_root, num_workers, size_batch, 
            num_classes_iter=num_classes_iter, num_elements_class=num_elements_class, 
            trans=trans, num_classes=num_classes, net_type=net_type, bssampling=bssampling)
    
    if os.path.basename(data_root) != 'In_shop':
        dl_ev, dl_ev_gnn = get_val_loaders(data_root, num_workers, size_batch, 
                num_classes_iter=num_classes_iter, num_elements_class=num_elements_class, 
                trans=trans, num_classes=num_classes, net_type=net_type, mode=mode)
        return dl_tr, dl_ev, None, dl_ev_gnn
    else:
        dl_gallery, dl_query, dl_ev_gnn = get_inshop_val_loader(data_root, num_workers, 
                trans=trans, num_classes=num_classes, net_type=net_type, mode=mode)
        
        return dl_tr, dl_query, dl_gallery, dl_ev_gnn

def get_train_loaders(data_root, num_workers, size_batch, num_classes_iter=None,
                   num_elements_class=None, trans='norm', num_classes=None, 
                   net_type='resnet50', bssampling=None):
    # Train Dataset
    if os.path.basename(data_root) != 'In_shop':
        Dataset = dataset.Birds_DML(
                    root=data_root,
                    labels=list(range(0, num_classes)),
                    transform=trans,
                    net_type=net_type)
    else:
       Dataset = dataset.Inshop_Dataset(
                    root = data_root,
                    mode = 'train',
                    transform = trans,
                    net_type=net_type) 

    list_of_indices_for_each_class = get_list_of_inds(Dataset)

    sampler = CombineSampler(list_of_indices_for_each_class,
                            num_classes_iter, num_elements_class,
                            batch_sampler=bssampling)
    drop_last = True

    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=size_batch,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True)
    
    return dl_tr


def get_val_loaders(data_root, num_workers, size_batch, num_classes_iter=None,
                   num_elements_class=None, trans='norm', magnitude=15,
                   number_aug=0, num_classes=None, net_type='resnet50', mode='train'):
    # Evaluation Dataset
    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dataset_ev = dataset.Birds_DML(
        root=data_root,
        labels=list(range(num_classes, class_end)),
        transform=trans,
        eval_reid=True,
        net_type=net_type)

    if 'gnn' in mode.split('_'):
        
        list_of_indices_for_each_class = get_list_of_inds(dataset_ev)
        sampler = CombineSampler(list_of_indices_for_each_class,
                                 num_classes_iter, num_elements_class)

        dl_ev = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        dl_ev_gnn = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=size_batch,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True)

    elif 'pseudo' in mode.split('_'):
        
        sampler = KReciprocalSampler(1, 7) #(num_classes_iter, num_elements_class)

        dl_ev_gnn = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=7, #size_batch,
            shuffle=False,
            sampler=sampler,
            num_workers=1,
            drop_last=True,
            pin_memory=True)

        dl_ev = torch.utils.data.DataLoader(
            copy.deepcopy(dataset_ev),
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

    else:
        dl_ev = torch.utils.data.DataLoader(
            dataset_ev,
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        dl_ev_gnn = None
    
    return dl_ev, dl_ev_gnn


def get_inshop_val_loader(data_root, num_workers, trans='norm', 
        num_classes=None, net_type='resnet50', mode='train'):

    query_dataset = dataset.Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = trans,
            net_type=net_type,
            eval_reid=True)
    
    dl_query = torch.utils.data.DataLoader(
            query_dataset,
            batch_size = 150,
            shuffle = False,
            num_workers = 4,
            pin_memory = True)
    
    gallery_dataset = dataset.Inshop_Dataset(
                root = data_root,
                mode = 'gallery',
                transform = trans,
                net_type=net_type,
                eval_reid=True)
    
    dl_gallery = torch.utils.data.DataLoader(
            gallery_dataset,
            batch_size = 150,
            shuffle = False,
            num_workers = 4,
            pin_memory = True)

    if 'pseudo' in mode.split('_'):
        import copy
        gnn_dataset = copy.deepcopy(query_dataset)
        gnn_dataset.im_paths.extend(gallery_dataset.im_paths)
        gnn_dataset.ys.extend(gallery_dataset.ys)
        
        sampler = KReciprocalSamplerInshop(1, 4)
        #sampler.path_to_ind = gnn_dataset.path_to_ind

        dl_ev_gnn = torch.utils.data.DataLoader(
            gnn_dataset,
            sampler=sampler,
            batch_size = 4,
            shuffle = False,
            num_workers = 4,
            pin_memory = True)
    else:
        dl_ev_gnn = None    
    
    return dl_gallery, dl_query, dl_ev_gnn


def get_list_of_inds(Dataset):
    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])
    return list_of_indices_for_each_class


def show_dataset(img, y):
    for i in range(img.shape[0]):
        im = img[i, :, :, :].squeeze()
        x = im.numpy().transpose((1, 2, 0))
        plt.imshow(x)
        plt.axis('off')
        plt.title('Image of label {}'.format(y[i]))
        plt.show()



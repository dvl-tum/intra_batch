from torch.utils.data.sampler import Sampler
import random
import copy
import torch
import sklearn.metrics.pairwise
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger('GNNReID.CombineSampler')

class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl, batch_sampler=None):
        logger.info("Combine Sampler")
        self.l_inds = l_inds
        self.max = -1
        self.cl_b = cl_b
        self.n_cl = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.feature_dict = None
        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)
        
        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(cl_b, n_cl)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()

        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:] 
            assert len(inds) == 0
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)


class NumberSampler():
    def __init__(self, num_classes, num_samples, seed=None):
        self.bs = num_classes * num_samples
        self.possible_denominators = [i for i in range(2, int(self.bs/2+1)) if self.bs%i == 0]
        seed = random.randint(0, 100) if seed is None else seed
        #seed = 4
        random.seed(seed)
        logger.info("Using seed {}".format(seed))

    def sample(self):
        num_classes = random.choice(self.possible_denominators)
        num_samples = int(self.bs/num_classes)
        logger.info("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples


class BatchSizeSampler():
    def __init__():
        seed = random.randint(0, 100)
        random.seed(seed)
        logger.info("Using seed {}".format(seed))
    def sample(self):
        num_classes = random.choice(range(2, 20))
        num_samples = random.choice(range(2, 20))
        logger.info("Number classes {}, number samples per class {}".format(num_classes, num_samples))
        return num_classes, num_samples

class KReciprocalSampler(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.k1 = 30
        self.k2 = self.bs

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):

        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.bs = self.num_classes * self.num_samples
            quality_checker.num_samples = self.bs
        
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            y = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            y =  torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            indices = [k for k in self.feature_dict.keys()]

        # generate distance mat for all classes as in Hierachrical Triplet Loss
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        dist = dist.cpu().numpy()
        sorted_dist = np.argsort(dist, axis=1)
        batches = list()
        exp = 0
        no = 0
        for i in range(sorted_dist.shape[0]):
            e = 0
            forward = sorted_dist[i, :self.k1 + 1]
            backward = sorted_dist[forward, :self.k1 + 1]
            rr = np.where(backward == i)[0]
            reciprocal = forward[rr]
            reciprocal_expansion = reciprocal
            for cand in reciprocal:
                cand_forward = sorted_dist[cand, :int(np.around(self.k1 / 2)) + 1]
                cand_backward = sorted_dist[cand_forward, :int(np.around(self.k1 / 2)) + 1]
                fi_cand = np.where(cand_backward == cand)[0]
                cand_reciprocal = cand_forward[fi_cand]
                if len(np.intersect1d(cand_reciprocal, reciprocal)) > 2 / 3 * len(
                        cand_reciprocal):
                    reciprocal_expansion = np.append(reciprocal_expansion, cand_reciprocal)
                    e =1
            if e == 1:
                exp +=1
            else: 
                no +=1
            reciprocal_expansion = np.unique(reciprocal_expansion)
            batch = reciprocal_expansion[np.argsort(dist[i, reciprocal_expansion])[:self.bs]].tolist()
            k = 0
            while len(batch) < self.bs:
                if sorted_dist[i, k] not in batch:
                    batch.append(sorted_dist[i, k])
                k += 1
            batch = [indices[k] for k in batch]
            assert len(batch) == self.bs
            batches.append(batch)
        
        random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class KReciprocalSamplerInshop(Sampler):
    def __init__(self, num_classes, num_samples, batch_sampler=None):
        # kNN
        self.feature_dict_query = None
        self.feature_dict_gallery = None
        self.double_gallery = None
        self.double_query = None
        self.bs = num_classes * num_samples
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.k1 = 30
        self.k2 = self.bs

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None

    def __iter__(self):
        
        num_query = len(self.feature_dict_query)
        
        if self.sampler:
            self.num_classes, self.num_samples = self.sampler.sample()
            self.bs = self.num_classes * self.num_samples
            quality_checker.num_samples = self.bs

        if type(self.feature_dict_query[list(self.feature_dict_query.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict_query.keys() for f in self.feature_dict_query[k].values()], 0)
            y = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict_gallery.keys() for f in self.feature_dict_gallery[k].values()], 0)
            self.labels_x = [k for k in self.feature_dict_query.keys() for f in self.feature_dict_query[k].values()]
            self.labels_y = [k for k in self.feature_dict_gallery.keys() for f in self.feature_dict_gallery[k].values()]
            indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict_query.values()], 0)
            y =  torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict_gallery.values()], 0)
            indices_x = [k for k in self.feature_dict_query.keys()]
            indices_y = [k for k in self.feature_dict_gallery.keys()]
        print("min max")
        print(min(indices_x), max(indices_x), min(indices_y), max(indices_y))        
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())
        dist = dist.cpu().numpy()
        sorted_dist = np.argsort(dist, axis=1)
        
        m, n = n, m
        dist_backward = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist_backward.addmm_(1, -2, y, x.t())
        dist_backward = dist_backward.cpu().numpy()
        sorted_dist_backward = np.argsort(dist_backward, axis=1)
        
        m, n = m, m
        dist_qq = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist_qq.addmm_(1, -2, y, y.t())
        dist_qq = dist_qq.cpu().numpy()
        sorted_dist_qq = np.argsort(dist_qq, axis=1)

        batches = list()
        
        for i in range(sorted_dist.shape[0]):
            
            forward = sorted_dist[i, :self.k1 + 1]
            #print("Dist")
            #print(self.double_query[i])
            #for ind in forward:
            #    print(self.double_gallery[ind])
            
            backward = sorted_dist_backward[forward, :self.k1 + 1]
            rr = np.where(backward == i)[0]
            reciprocal = forward[rr]
            reciprocal_expansion = reciprocal
            for cand in reciprocal:
                cand_forward = sorted_dist_qq[cand, :int(np.around(self.k1 / 2)) + 1]
                cand_backward = sorted_dist_qq[cand_forward, :int(np.around(self.k1 / 2)) + 1]
                fi_cand = np.where(cand_backward == cand)[0]
                cand_reciprocal = cand_forward[fi_cand]
                if len(np.intersect1d(cand_reciprocal, reciprocal)) > 2 / 3 * len(
                        cand_reciprocal):
                    #print(reciprocal_expansion)
                    reciprocal_expansion = np.append(reciprocal_expansion, cand_reciprocal)
            #print(reciprocal_expantion)
            reciprocal_expansion = np.unique(reciprocal_expansion)
            #print(reciprocal_expansion)
            #print(dist[i, reciprocal_expansion])
            batch = reciprocal_expansion[np.argsort(dist[i, reciprocal_expansion])[:self.bs-1]].tolist()
            k = 0
            while len(batch) < self.bs:
                if sorted_dist[i, k] not in batch:
                    batch.append(sorted_dist[i, k])
                k += 1
            #print("batch")
            #print(self.double_query[i])
            #for ind in batch[:self.bs-1]:
            #    print(ind, self.double_gallery[ind])
            batch = [indices_x[i]] + [indices_y[k] for k in batch[:self.bs-1]]
            #print()

            assert len(batch) == self.bs
            batches.append(batch)

        #random.shuffle(batches)
        self.flat_list = [s for batch in batches for s in batch]
        return (iter(self.flat_list))

    def __len__(self):
        return len(self.flat_list)


class ClusterSampler(Sampler):
    def __init__(self, num_classes, num_samples, nb_clusters=None, batch_sampler=None):
        # kmeans
        self.feature_dict = None
        self.bs = num_classes * num_samples
        self.cl_b = num_classes
        self.n_cl = num_samples
        self.epoch = 0
        self.nb_clusters = nb_clusters

        if batch_sampler == 'NumberSampler':
            self.sampler = NumberSampler(num_classes, num_samples)
        elif batch_sampler == 'BatchSizeSampler':
            self.sampler = BatchSizeSampler()
        else:
            self.sampler = None
        
    def get_clusters(self):
        logger.info(self.nb_clusters)
        # generate distance mat for all classes as in Hierachrical Triplet Loss
        if type(self.feature_dict[list(self.feature_dict.keys())[0]]) == dict:
            x = torch.cat([f.unsqueeze(0).cpu() for k in self.feature_dict.keys() for f in self.feature_dict[k].values()], 0)
            self.labels = [k for k in self.feature_dict.keys() for f in self.feature_dict[k].values()]
            self.indices = [ind for k in self.feature_dict.keys() for ind in self.feature_dict[k].keys()]
        else:
            x = torch.cat([f.unsqueeze(0).cpu() for f in self.feature_dict.values()], 0)
            self.indices = [k for k in self.feature_dict.keys()]
        self.nb_clusters = 900
        logger.info("Kmeans")
        self.cluster = sklearn.cluster.KMeans(self.nb_clusters).fit(x).labels_        
        #logger.info('spectral')
        #self.cluster = sklearn.cluster.SpectralClustering(self.nb_clusters, assign_labels="discretize", random_state=0).fit(x).labels_
        #self.nb_clusters = 600
        #logger.info('ward')
        #self.cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=self.nb_clusters).fit(x).labels_
        #logger.info('DBSCAN')
        #eps = 0.9
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x).labels_
        #logger.info("Optics")
        #eps = 0.9
        #min_samples = 5
        #logger.info("Eps {}, min samples {}".format(eps, min_samples))
        #self.cluster = sklearn.cluster.OPTICS(min_samples=min_samples, eps=eps).fit(x).labels_
        #logger.info("Birch")
        #self.cluster = sklearn.cluster.Birch(n_clusters=self.nb_clusters).fit(x).labels_

    def __iter__(self):
        if self.sampler:
            self.cl_b, self.n_cl = self.sampler.sample()
            quality_checker.num_samps=self.n_cl
        if self.epoch % 5 == 2:
            self.get_clusters()
        
        ddict = defaultdict(list)
        for idx, label in zip(self.indices, self.cluster):
            ddict[label].append(idx)

        l_inds = []
        for key in ddict:
            l_inds.append(ddict[key]) 
        
        l_inds = list(map(lambda a: random.sample(a, len(a)), l_inds))

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.n_cl:
                inds += [random.choice(choose)]

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.n_cl + 1)*self.n_cl - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                self.quality_checker.check([self.labels[i] for i in inds[:self.n_cl]], inds[:self.n_cl])
                inds = inds[self.n_cl:] 
            assert len(inds) == 0
                
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.cl_b != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.cl_b - len(split_list_of_indices) % self.cl_b, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        assert len(split_list_of_indices) % self.cl_b == 0

        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]

        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)



from . import calc_normalized_mutual_information, cluster_by_kmeans, \
        assign_by_euclidian_at_k, calc_recall_at_k, \
        assign_by_cos_sim_InSop, calc_recall_at_k_InShop, assign_by_euclidean_at_k_InShop
import torch
import logging
import json
from collections import defaultdict
import sklearn.cluster
import sklearn.metrics.cluster
import os
import time
logger = logging.getLogger('GNNReID.Evaluator')


class Evaluator_DML():
    def __init__(self, output_test_enc='norm', output_test_gnn='norm', cat=0, nb_clusters=0, dev=0):
        
        self.nb_clusters = nb_clusters
        self.output_test_enc = output_test_enc
        self.output_test_gnn = output_test_gnn
        self.cat = cat
        self.dev = dev

    def evaluate(self, model, dataloader, gallery_dl,
            gnn=None, graph_generator=None, dl_ev_gnn=None, net_type='bn_inception',
            dataroot='CARS', nb_classes=None):
        self.dataroot = dataroot
        
        self.gallery_dl = gallery_dl
        self.nb_classes = nb_classes
        start = time.time()
        model_is_training = model.training
        model.eval()
        
        # calculate embeddings with model, also get labels (non-batch-wise)
        X, T, P = self.predict_batchwise(model, dataloader)
        if dataroot == 'in_shop':
            gallery_X, gallery_T, gallery_P = self.predict_batchwise(model, gallery_dl)
        
        mode = self.get_mode(dl_ev_gnn, dataloader)

        if mode is not 'backbone' and dataroot != 'in_shop':
            gnn_is_training = gnn.training
            gnn.eval()
            logger.info("Evaluate KNN evaluate")
            X, T, P = self.predict_batchwise_gnn(gnn, graph_generator, X, T, P, dl_ev_gnn, mode)
            gnn.train(gnn_is_training)

        elif dl_ev_gnn is not None and dataroot == 'in_shop':
            X, T, P = self.predict_batchwise_gnn_inshop(gnn, graph_generator, dl_ev_gnn, gallery_X, gallery_T, gallery_P, X, T, P)
        end = time.time()
        logger.info("Time for feature vector computation {}".format(end-start)) 
        
        if  dataroot != 'in_shop'and dataroot != 'sop':
            # calculate NMI with kmeans clustering
            nmi = calc_normalized_mutual_information(T, cluster_by_kmeans(X, nb_classes))
            logger.info("NMI: {:.3f}".format(nmi * 100))
        else:
            nmi = -1
        
        # Get Recall
        recall = []
        if dataroot != 'sop' and dataroot != 'in_shop':
            Y, T = assign_by_euclidian_at_k(X, T, 8)  # assign_by_cos_sim(X, T, 8)
            which_nearest_neighbors = [1, 2, 4, 8]

        elif dataroot == 'in_shop':
            #Y, T = assign_by_euclidean_at_k_InShop(X, gallery_X, gallery_T, T, 30)
            cos_sim = assign_by_cos_sim_InSop(X, gallery_X)
            which_nearest_neighbors = [1, 10, 20, 30]

        else:
            Y, T = assign_by_euclidian_at_k(X, T, 1000)
            which_nearest_neighbors = [1, 10, 100, 1000]
        
        if dataroot != 'in_shop':
            for k in which_nearest_neighbors:
                r_at_k = calc_recall_at_k(T, Y, k)
                recall.append(r_at_k)
                logger.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
        else:
            for k in which_nearest_neighbors:
                #r_at_k = calc_recall_at_k(T, Y, k)
                r_at_k =  calc_recall_at_k_InShop(cos_sim, T, gallery_T, k)
                recall.append(r_at_k)
                logger.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

        model.train(model_is_training)  # revert to previous training state
        return nmi, recall

    # just looking at this gives me AIDS, fix it fool!
    def predict_batchwise(self, model, dataloader):
        logger.info("Evaluate normal")
        paths = []
        fc7s, Ys = list(), list()
        with torch.no_grad():
            for X, Y, I, P in dataloader:
                if torch.cuda.is_available(): X = X.to(self.dev)
                _, fc7 = model(X, output_option=self.output_test_enc, val=True)
                
                fc7s.append(fc7)
                Ys.append(Y)
                paths.append(P)
                
        fc7 = torch.cat([f.unsqueeze(0).cpu() for b in fc7s for f in b], 0)
        Y = torch.cat([y.unsqueeze(0).cpu() for b in Ys for y in b], 0)
        paths = [p for b in paths for p in b]
        
        return torch.squeeze(fc7), torch.squeeze(Y), paths

    def predict_batchwise_gnn(self, gnn, graph_generator, X, T, P, dl_ev_gnn, mode, X_G=None, P_G=None):
        logger.info("KNN")

        in_shop = True if X_G is not None else False
        
        if self.dataroot != 'sop':
            self.get_resnet_performance(X, T)
        
        # Update after feature dict for sampling
        pti = dl_ev_gnn.dataset.path_to_ind
        feature_dict = {pti[p]: f for p, f in zip(P, X)}
        gallery_dict = {pti[p]: f for p, f in zip(P_G, X_G)} if in_shop else None
        dl_ev_gnn.sampler.feature_dict = feature_dict
        
        features, labels, paths = list(), list(), list()
        with torch.no_grad():
            for X, Y, I, P in dl_ev_gnn:
                fc7 = torch.stack([feature_dict[i.data.item()] for i in I]).to(self.dev)

                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
                
                _, fc7 = gnn(fc7, edge_index, edge_attr,
                             output_option=self.output_test_gnn)
                
                if self.cat:
                    fc7 = torch.cat(fc7, dim=1)
                else:
                    fc7 = fc7[-1]
                
                if mode == 'gnn':
                    labels.append(Y)
                    features.append(fc7)
                    paths.append(P)
                else:
                    labels.append(Y[0])
                    features.append(fc7[0])
                    paths.append(P[0])
        
        if mode == 'gnn':
            features = [f.unsqueeze(dim=0).cpu() for b in features for f in b]
            labels = [l.unsqueeze(dim=0).cpu() for b in labels for l in b]
            paths = [p for b in paths for p in b]
            # filter out double samples
            features = torch.cat([features[i-1] for i in range(1, len(paths)+1) if paths[i-1] not in paths[:i-1]]).squeeze()
            labels = torch.cat([labels[i-1] for i in range(1, len(paths)+1) if paths[i-1] not in paths[:i-1]]).squeeze()
            paths = [paths[i-1] for i in range(1, len(paths)+1) if paths[i-1] not in paths[:i-1]]
        else:
            features = torch.cat([f.unsqueeze(dim=0).cpu() for f in features]).squeeze()
            labels = torch.cat([l.unsqueeze(dim=0).cpu() for l in labels]).squeeze()
        
        return features, labels, paths

    def predict_batchwise_gnn_inshop(self, gnn, graph_generator, dl_ev_gnn, gallery_X, gallery_T, gallery_P, X, T, P):
        import copy
        gallery = copy.deepcopy(gallery_X)
        dl_ev_gnn.sampler.feature_dict_gallery = {i+len(X): f for i, f in enumerate(gallery)}
        dl_ev_gnn.sampler.feature_dict_query = {i: f for i, f in enumerate(X)}
        dl_ev_gnn.sampler.double_query = {i: [p, y] for i, p, y in zip(range(len(P)), P, T)}
        dl_ev_gnn.sampler.double_gallery = {i: [p, y] for i, p, y in zip(range(len(gallery_P)), gallery_P, gallery_T)}

        features_new = dict()
        labels_new = dict()

        model_is_training = gnn.training
        gnn.eval()
        query_T, query_X, query_P = dict(), dict(), list()
        num_query = len(X)
        
        with torch.no_grad():
            for batch in dl_ev_gnn:
                _, Y, I, P = batch
                print(Y)
                fc7 = torch.stack([X[I[0]]] + [gallery[i-num_query] for i in I[1:]]).cuda()

                edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)

                _, fc7 = gnn(fc7, edge_index, edge_attr,
                            output_option='plain')

                query_T[P[0]] = Y[0]
                query_X[P[0]] = fc7[-1][0]
                query_P.append(P)

        query_X = torch.stack(list(query_X.values())).cpu()
        query_T = torch.stack(list(query_T.values())).cpu()
        query_P = [p for b in query_P for p in b]
        
        return query_X, query_T, query_P
    
    def get_resnet_performance(self, x, ys):

        cluster = sklearn.cluster.KMeans(self.nb_classes).fit(x).labels_
        NMI = sklearn.metrics.cluster.normalized_mutual_info_score(cluster, ys)
        logger.info("KNN: NMI after ResNet50 {}".format(NMI))

        RI = sklearn.metrics.adjusted_rand_score(ys, cluster)
        logger.info("RI after Resnet50 {}".format(RI))

        Y, ys_ = assign_by_euclidian_at_k(x, ys, 1)
        r_at_k = calc_recall_at_k(ys_, Y, 1)
        logger.info("KNN: R@{} after ResNet50: {:.3f}".format(1, 100 * r_at_k))
    
    def get_mode(self, dl_ev_gnn, dataloader):
        if dl_ev_gnn is not None:
            if dl_ev_gnn.sampler.__class__.__name__ == 'CombineSampler':
                mode = 'gnn'
            else:
                mode = 'rknn'
        else:
            mode = 'backbone'
        return mode

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    query_X, query_T, query_p = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T, gallery_p = predict_batchwise(model, gallery_dataloader)

    query_dict = {p: f for p, f in zip(query_p, query_X)}
    gallery_dict = {p: f for p, f in zip(gallery_p, gallery_X)}

    recall = get_recall(query_X, gallery_X, gallery_T, query_T)

    return recall, query_dict, gallery_dict, gallery_T


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * paths
                if i == 0:
                    # move images to device of model (approximate device)
                    _, J = model(J.cuda())


                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state

    return torch.stack(A[0]), torch.stack(A[1]), A[2]


def get_recall(query_X, gallery_X, gallery_T, query_T):
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    import torch.nn.functional as F 
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1

        return match_counter / m

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    quit()
    return recall

def evaluate_gnn_inshop(gnn, graph_generator, dl_ev_gnn, gallery_X, gallery_T, gallery_P, X, T, P):

    dl_ev_gnn.sampler.feature_dict_gallery = {i+len(X)-1: f for i, f in enumerate(gallery_X)}
    dl_ev_gnn.sampler.feature_dict_query = {i: f for i, f in enumerate(X)}

    features_new = dict()
    labels_new = dict()

    query_T, query_X = predict_batchwise_gnn_inshop(gnn, graph_generator, dl_ev_gnn, X, gallery_X)

    query_X = torch.stack(list(query_X.values())).cpu()
    query_T = torch.stack(list(query_T.values())).cpu()
    
    get_recall(query_X, gallery_X, gallery_T.cpu(), query_T)

def predict_batchwise_gnn_inshop(gnn, graph_generator, dataloader, query_features, gallery_features):
    model_is_training = gnn.training
    gnn.eval()
    labels, features = dict(), dict()
    num_query = len(query_features)

    with torch.no_grad():
        for batch in dataloader:
            _, Y, I, P = batch
            fc7 = torch.stack([query_features[I[0]]] + [gallery_features[i-num_query+1] for i in I[1:]]).cuda()

            edge_attr, edge_index, fc7 = graph_generator.get_graph(fc7)
            #fc7 = fc7.cuda()
            #edge_attr = edge_attr.cuda()
            #edge_index = edge_index.cuda()

            _, fc7 = gnn(fc7, edge_index, edge_attr,
                            output_option='plain')

            labels[P[0]] = Y[0]
            features[P[0]] = fc7[-1][0]

    return labels, features


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


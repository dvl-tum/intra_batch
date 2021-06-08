import torch
import torch.nn.functional as F
import numpy as np
import sklearn.metrics.pairwise


def assign_by_euclidean_at_k_InShop(query_X, gallery_X, gallery_T, query_T, k):

    distances = sklearn.metrics.pairwise_distances(query_X.cpu(), gallery_X.cpu(), 'cosine')
    indices = np.argsort(distances, axis = 1)[:, 0 : k]

    labs = np.array([[gallery_T[i] for i in ii] for ii in indices])

    return labs, query_T


def assign_by_cos_sim_InSop(query_X, gallery_X):
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    cos_sim = F.linear(query_X, gallery_X)

    return cos_sim


def calc_recall_at_k_InShop(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1

        return match_counter / m


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


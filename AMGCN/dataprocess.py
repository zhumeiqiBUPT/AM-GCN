import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from utils import normalize

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def process_data(dataset):
    names = ['y', 'ty', 'ally','x', 'tx', 'allx','graph']
    objects = []
    for i in range(len(names)):
        with open("../data/cache/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty, ally, x, tx, allx, graph = tuple(objects)
    print(graph)
    test_idx_reorder = parse_index_file("../data/cache/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    print(features)
    f = open('../data/{}/{}.adj'.format(dataset, dataset), 'w+')
    for i in range(len(graph)):
        adj_list = graph[i]
        for adj in adj_list:
            f.write(str(i) + '\t' + str(adj) + '\n')
    f.close()

    label_list = []
    for i in labels:
        label = np.where(i == np.max(i))[0][0]
        label_list.append(label)
    np.savetxt('../data/{}/{}.label'.format(dataset, dataset), np.array(label_list), fmt='%d')
    np.savetxt('../data/{}/{}.test'.format(dataset, dataset), np.array(test_idx_range), fmt='%d')
    np.savetxt('../data/{}/{}.feature'.format(dataset, dataset), features, fmt='%f')


def construct_graph(dataset, features, topk):
    fname = '../data/' + dataset + '/knn/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset):
    for topk in range(2, 10):
        data = np.loadtxt('../data/' + dataset + '/' + dataset + '.feature', dtype=float)
        print(data)
        construct_graph(dataset, data, topk)
        f1 = open('../data/' + dataset + '/knn/tmp.txt','r')
        f2 = open('../data/' + dataset + '/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()

''' process cora/citeseer/pubmed data '''
#process_data('citeseer')

'''generate KNN graph'''

#generate_knn('uai')


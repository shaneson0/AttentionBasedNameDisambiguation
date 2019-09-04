from os.path import join
import numpy as np
import scipy.sparse as sp
from utils import settings
from global_.prepare_local_data import IDF_THRESHOLD
import json
import codecs

local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESHOLD))

def loadJson(path):
    with open(path, 'r') as fp:
        result = json.load(fp)
        fp.close()
        return result

def getAuthor2Id():
    return loadJson('./data/global_/Author2Id.json')

def Id2Author():
    return loadJson('./data/global_/Id2Author.json')

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))

def buildGraph(graphPath, idx_features_labels, features):
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(graphPath, dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

def load_local_data(path=local_na_dir, name='cheng_cheng'):
    # Load local paper network dataset
    print('Loading {} dataset...'.format(name), 'path=', path)

    idx_features_labels = np.genfromtxt(join(path, "{}_pubs_content.txt".format(name)), dtype=np.dtype(str))
    print(idx_features_labels)
    features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?

    # print(list(set(idx_features_labels[:, -1])))
    # print('len of cluster label', len(list(set(idx_features_labels[:, -1]))))


    labels = encode_labels(idx_features_labels[:, -2])
    Clusterlabels = encode_labels(idx_features_labels[:, -1])

    # build graph
    adj = buildGraph(join(path, "{}_pubs_network.txt".format(name)), idx_features_labels, features)
    adj2 = buildGraph(join(path, "{}_pubs_network2.txt".format(name)), idx_features_labels, features)

    # print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return adj, adj2, features, labels, Clusterlabels

def loadAuthorSocial():
    AuthorSocial = {}
    with codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_social.txt'), 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            items = line.rstrip().split('\t')
            # print(items[0], ' ', list(map(lambda x: int(x) , items[1].split(' '))))
            AuthorSocial[items[0]] = list(map(lambda x: int(x) , items[1].split(' ')))
    return AuthorSocial

if __name__ == '__main__':
    adj, features, labels = load_local_data(name='hongbin_li')
    loadAuthorSocial()
    # print(adj.shape)
    # print(len(labels))
    # print(len(set(labels)))














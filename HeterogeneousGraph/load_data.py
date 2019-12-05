
import numpy as np
from os.path import join
from utils import settings, string_utils
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
IDF_THRESHOLD = 32  # small data
# IDF_THRESHOLD = 10

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    res = [[label, classes_dict[label]] for label in labels]
    return enc.fit_transform(res).toarray()

def getPATH(name, idf_threshold):
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    path = join(graph_dir, '{}_feature_and_label.txt'.format(name))
    return path

def loadFeature(name, idf_threshold=IDF_THRESHOLD):
    featurePath = getPATH(name, idf_threshold)
    # idx_features_labels = np.genfromtxt(join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)), dtype=np.dtype(str))
    idx_features_labels = np.genfromtxt(featurePath, dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?
    labels = encode_labels(idx_features_labels[:, -2])
    return features, labels

def loadPAP(name, idf_threshold=IDF_THRESHOLD):
    PAPPATH = getPATH(name, idf_threshold)
    PAP = np.genfromtxt(PAPPATH, dtype=np.dtype(str))
    return PAP

def loadData(name, idf_threshold=32):
    pass
    # graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    # PAP = open(join(graph_dir, '{}_PAP.txt'.format(name)), 'w')
    # PSP = open(join(graph_dir, '{}_PSP.txt'.format(name)), 'w')
    # feature = open(join(graph_dir, '{}_feature_and_label.txt'.format(name)), 'w')
    # # build graph
    # adj = buildGraph(join(path, "{}_pubs_network.txt".format(name)), idx_features_labels, features)
    # adj2 = buildGraph(join(path, "{}_pubs_network2.txt".format(name)), idx_features_labels, features)
    # idx_features_labels = np.genfromtxt(path, dtype=np.dtype(str))
    #
    # p


if __name__ == '__main__':
    # kexin_xu is nulls
    name = 'zhigang_zeng'
    # loadData(name)
    # loadFeature(name)
    loadPAP(name)









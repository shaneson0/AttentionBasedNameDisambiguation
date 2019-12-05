
import numpy as np
from os.path import join
from utils import settings, string_utils

IDF_THRESHOLD = 32  # small data
# IDF_THRESHOLD = 10

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def loadFeature(name, idf_threshold=IDF_THRESHOLD):
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    feature = open(join(graph_dir, '{}_feature_and_label.txt'.format(name)), 'w')
    idx_features_labels = np.genfromtxt(feature, dtype=np.dtype(str))
    print (idx_features_labels)

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
    name = 'kexin_xu'
    # loadData(name)
    loadFeature(name)




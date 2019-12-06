
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

def constructAdj(pids):
    pid2idx = {c: i for i, c in enumerate(pids)}
    idx2pid = {i: c for i, c in enumerate(pids)}

    LenPids = len(pids)
    PAP = np.zeros(shape=( LenPids, LenPids ))
    PSP = np.zeros(shape=( LenPids, LenPids ))
    return PAP, PSP, pid2idx, idx2pid



def getPATH(name, idf_threshold, filename):
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    path = join(graph_dir, '{}_{}.txt'.format(name, filename))
    return path

def loadFeature(name, idf_threshold=IDF_THRESHOLD):
    featurePath = getPATH(name, idf_threshold, 'feature_and_label')
    # idx_features_labels = np.genfromtxt(join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)), dtype=np.dtype(str))
    idx_features_labels = np.genfromtxt(featurePath, dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?
    labels = encode_labels(idx_features_labels[:, -2])
    pids = idx_features_labels[:, 0]
    return features, labels, pids

def loadPAP(PAP, pid2idx, name, idf_threshold=IDF_THRESHOLD):
    PAPPATH = getPATH(name, idf_threshold, 'PAP')
    PAPPath = np.genfromtxt(PAPPATH, dtype=np.dtype(str))
    for _from, _to in PAPPath:
        PAP[pid2idx[_from]][pid2idx[_to]] = 1
        PAP[pid2idx[_to]][pid2idx[_from]] = 1
    return PAP

def loadPSP(PSP, pid2idx, name, idf_threshold=IDF_THRESHOLD):
    PSPPATH = getPATH(name, idf_threshold, 'PSP')
    PSPPath = np.genfromtxt(PSPPATH, dtype=np.dtype(str))
    for _from, _to in PSPPath:
        PSP[pid2idx[_from]][pid2idx[_to]] = 1
        PSP[pid2idx[_to]][pid2idx[_from]] = 1
    return PSP


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
    features, labels, pids = loadFeature(name)
    PAP, PSP, pid2idx, idx2pid = constructAdj(pids)

    PAP = loadPAP(PAP,pid2idx, name)
    print (PAP)
    print (PAP.tolist())

    PSP = loadPSP(PSP, pid2idx, name)
    print (PSP)
    print (PSP.tolist())









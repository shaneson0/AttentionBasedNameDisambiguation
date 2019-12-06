
import numpy as np
from os.path import join
from utils import settings, string_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from HeterogeneousGraph.HAN import HAN


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

def constructIdx(X):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=1)
    return X_train, X_val, X_test




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


def load_data_dblp(truelabels, truefeatures, PAP, PSP, train_idx, val_idx, test_idx):
    rownetworks = [PAP, PSP]

    y = truelabels

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask




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

    N = len(pids)
    X_train, X_val, X_test = constructIdx(list(range(N)))
    print (X_train, X_val, X_test)

    han = HAN()
    adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = han.load_data_dblp(labels, features, PAP, PSP, X_train, X_val, X_test)














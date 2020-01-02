from os.path import abspath, dirname, join
from utils.cache import LMDBClient
from utils import data_utils, settings, encode_labels, tSNEAnanlyse
from utils import clustering, pairwise_precision_recall_f1
import numpy as np

from utils import clustering, pairwise_precision_recall_f1

IDF_THRESHOLD = 10


def encode_labels(self, labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    res = [[label, classes_dict[label]] for label in labels]
    rawlabels = [classes_dict[label] for label in labels]
    return self.enc.fit_transform(res).toarray(), np.array(rawlabels)

def getPATH(name, idf_threshold, filename, ispretrain):
    if ispretrain:
        graph_dir = join(settings.DATA_DIR, 'AttentionNetwork', 'graph-{}'.format(idf_threshold))
    else:
        graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))

    path = join(graph_dir, '{}_{}.txt'.format(name, filename))
    return path

def loadFeature(name, idf_threshold=IDF_THRESHOLD, ispretrain=True):
    EndIndex = -1
    if ispretrain is False:
        EndIndex = -2
    featurePath = getPATH(name, idf_threshold, 'feature_and_label', ispretrain)
    # idx_features_labels = np.genfromtxt(join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)), dtype=np.dtype(str))
    idx_features_labels = np.genfromtxt(featurePath, dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1:EndIndex], dtype=np.float32)  # sparse?
    labels, rawlabels = encode_labels(idx_features_labels[:, EndIndex])
    pids = idx_features_labels[:, 0]
    return features, labels, pids, rawlabels

def load_test_names():
    return data_utils.load_json(settings.DATA_DIR, 'test_name_list2.json')

Res = {}

names = load_test_names()
for name in names:
    features, labels, pids, rawlabels = load_test_names()
    tSNEAnanlyse(features, rawlabels, join(settings.PIC_DIR, "MetricLearning", "rawReature_%s_train.png" % (name)))
    numberofLabels = len(set(rawlabels))
    clusters_pred = clustering(features, num_clusters=numberofLabels)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    Res[name] = {"prec": prec, "rec": rec, "f1": f1}


print (Res)

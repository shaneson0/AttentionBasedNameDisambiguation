
from os.path import abspath, dirname, join
from utils.cache import LMDBClient
from utils import data_utils, settings, encode_labels, tSNEAnanlyse
from utils import clustering, pairwise_precision_recall_f1


rawFeatureLMDBName = "author_100.emb.weighted"
rawFeature = LMDBClient(rawFeatureLMDBName)

tripleteLossLMDBName = 'author_triplets.emb'
tripletFeature = LMDBClient(tripleteLossLMDBName)

LMDB_NAME_EMB = "lc_attention_network_embedding2"
lc_emb = LMDBClient(LMDB_NAME_EMB)

def load_train_names():
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    return name_to_pubs_train


name_to_pubs_train= load_train_names()
# for name in name_to_pubs_train:

name = "gang_yin"
cur_author = name_to_pubs_train[name]
pids = []
labels = []
rf = []
tf = []
attentionf = []

for aid in cur_author:
    if len(cur_author[aid]) < 5:
        continue

    for pid in cur_author[aid]:
        pids.append(pid)
        labels.append(aid)
        rf.append(rawFeature.get(pid))
        tf.append(tripletFeature.get(pid))
        attentionf.append(lc_emb.get(pid))

labels = encode_labels(labels)
numberofLabels = len(set(labels))


def clusterTest(embedding, numberofLabels):
    clusters_pred = clustering(embedding, num_clusters=numberofLabels)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    return [prec, rec, f1]


tSNEAnanlyse(rf, labels, join(settings.PIC_DIR, "FINALResult", "%s_rawFeature.png" % (name)))
tSNEAnanlyse(tf, labels, join(settings.PIC_DIR, "FINALResult", "%s_tripletFeature.png" % (name)))
tSNEAnanlyse(attentionf, labels, join(settings.PIC_DIR, "FINALResult", "%s_lcmbFeature.png" % (name)))

Res = {}
Res['rawfeature'] = clusterTest(rf, numberofLabels=numberofLabels)
Res['tripletfeature'] = clusterTest(tf, numberofLabels=numberofLabels)
Res['lcmbfeature'] = clusterTest(attentionf, numberofLabels=numberofLabels)

print ("Res: ", Res)
























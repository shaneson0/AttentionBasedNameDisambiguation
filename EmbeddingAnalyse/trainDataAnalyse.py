
from os.path import abspath, dirname, join
from utils.cache import LMDBClient
from utils import data_utils, settings, encode_labels, tSNEAnanlyse

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
name = "yanjun_zhang"
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

tSNEAnanlyse(rf, labels, join(settings.PIC_DIR, "FINALResult", "%s_rawFeature.png" % (name)))
tSNEAnanlyse(tf, labels, join(settings.PIC_DIR, "FINALResult", "%s_tripletFeature.png" % (name)))
# tSNEAnanlyse(attentionf, labels, join(settings.PIC_DIR, "FINALResult", "%s_lcmbFeature.png" % (name)))

res = lc_emb.get('5b5433eee1cd8e4e150b7d98-1')
print ("res test: ", res)























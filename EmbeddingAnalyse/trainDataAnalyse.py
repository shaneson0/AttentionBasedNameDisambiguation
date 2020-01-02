
from os.path import abspath, dirname, join
from utils.cache import LMDBClient
from utils import data_utils, settings, encode_labels, tSNEAnanlyse
from utils import clustering, pairwise_precision_recall_f1


rawFeatureLMDBName = "author_100.emb.weighted"
rawFeature = LMDBClient(rawFeatureLMDBName)

tripleteLossLMDBName = 'author_triplets.emb'
tripletFeature = LMDBClient(tripleteLossLMDBName)

# LMDB_NAME_EMB = "lc_attention_network_embedding2"
LMDB_NAME_EMB = "triplete_loss_lc_attention_network_embedding"
lc_emb = LMDBClient(LMDB_NAME_EMB)

LMDB_NAME_EMB = "raw_transform_local_embedding"
transferEmbedding = LMDBClient(LMDB_NAME_EMB)

# LMDB_NAME_EMB = "triplete_loss_lc_attention_network_embedding"
# lc_triplet_emb = LMDBClient(LMDB_NAME_EMB)

def load_train_names():
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    return name_to_pubs_train

def load_test_names():
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    return name_to_pubs_test


# name_to_pubs_train= load_train_names()
name_to_pubs_test= load_test_names()
# for name in name_to_pubs_train:

# name = "hai_yan_chen"
# name = "gang_yin"
# name = "hongbin_li" #Res:  {'rawfeature': [0.6501069645828381, 0.5840273329062566, 0.6152980877390326], 'tripletfeature': [0.6968630900029317, 0.5075806107196241, 0.5873486533234493], 'lcmbfeature': [0.7999494821924729, 0.6762758915225283, 0.7329321916223098], 'transferfeature': [0.3861209964412811, 0.27802690582959644, 0.3232774674115456]}
name = "kexin_xu"


cur_author = name_to_pubs_test[name]
pids = []
labels = []
rf = []
tf = []
attentionf = []
transferF = []

for aid in cur_author:
    if len(cur_author[aid]) < 5:
        continue

    for pid in cur_author[aid]:
        pids.append(pid)
        labels.append(aid)
        rf.append(rawFeature.get(pid))
        tf.append(tripletFeature.get(pid))
        attentionf.append(lc_emb.get(pid))
        transferF.append(transferEmbedding.get(pid))

labels = encode_labels(labels)
numberofLabels = len(set(labels))


def clusterTest(embedding, numberofLabels):
    clusters_pred = clustering(embedding, num_clusters=numberofLabels)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    return [prec, rec, f1]


tSNEAnanlyse(rf, labels, join(settings.PIC_DIR, "FINALResult", "%s_rawFeature.png" % (name)))
tSNEAnanlyse(tf, labels, join(settings.PIC_DIR, "FINALResult", "%s_tripletFeature.png" % (name)))
tSNEAnanlyse(attentionf, labels, join(settings.PIC_DIR, "FINALResult", "%s_lcmbFeature.png" % (name)))
tSNEAnanlyse(transferF, labels, join(settings.PIC_DIR, "FINALResult", "%s_transferFeature.png" % (name)))

Res = {}
Res['rawfeature'] = clusterTest(rf, numberofLabels=numberofLabels)
Res['tripletfeature'] = clusterTest(tf, numberofLabels=numberofLabels)
Res['lcmbfeature'] = clusterTest(attentionf, numberofLabels=numberofLabels)
Res['transferfeature'] = clusterTest(transferF, numberofLabels=numberofLabels)

print ("Res: ", Res)







# gang_yin
# Res:  {'rawfeature': [0.6346938775510204, 0.3823770491803279, 0.47723785166240407], 'tripletfeature': [0.8391248391248392, 0.5344262295081967, 0.6529794692038057], 'lcmbfeature': [0.9170305676855895, 0.860655737704918, 0.8879492600422833]}

# hai_yan_chen
# Res:  {'rawfeature': [0.41139954853273136, 0.2987704918032787, 0.34615384615384615], 'tripletfeature': [0.5184397163120568, 0.2995901639344262, 0.37974025974025977], 'lcmbfeature': [0.8192771084337349, 0.4737704918032787, 0.6003635419371592]}














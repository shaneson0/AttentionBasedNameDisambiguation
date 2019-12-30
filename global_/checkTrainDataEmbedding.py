from utils.cache import LMDBClient
from utils import data_utils, inputData, encode_labels, tSNEAnanlyse
from utils import settings
from global_.global_model import GlobalTripletModel
import numpy as np
from utils.eval_utils import get_hidden_output
from os.path import abspath, dirname, join

LMDB_NAME = "author_100.emb.weighted"
lc_input = LMDBClient(LMDB_NAME)
global_model = GlobalTripletModel(data_scale=1000000)
trained_global_model = global_model.load_triplets_model()
name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')

names = ["gang_yin", "gang_zou", "guo_hua_zhang", "h_hu", "hai_yan_chen", "hai_yang_li"]

for name in names:
    name_data = name_to_pubs_train[name]

    res_embs = []
    embs_input = []
    labels = []
    pids = []
    for i, aid in enumerate(name_data.keys()):
        if len(name_data[aid]) < 5:  # n_pubs of current author is too small
            continue
        for pid in name_data[aid]:
            cur_emb = lc_input.get(pid)
            if cur_emb is None:
                continue
            embs_input.append(cur_emb)
            pids.append(pid)
            labels.append(aid)


    embs_input = np.stack(embs_input)
    inter_embs = get_hidden_output(trained_global_model, embs_input)
    labels = encode_labels(labels)

    for i, pid_ in enumerate(pids):
        res_embs.append(inter_embs[i])

    # Clustering and save the result
    tSNEAnanlyse(res_embs, labels, join(settings.PIC_DIR, "OnlyTriplete", "rawReature_%s_triplet.png" % (name)))
    tSNEAnanlyse(embs_input, labels, join(settings.PIC_DIR, "OnlyTriplete", "rawReature_%s_features.png" % (name)))








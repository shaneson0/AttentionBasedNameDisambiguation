
from HeterogeneousGraph.HAN import HAN
from utils import settings
from utils import data_utils, eval_utils
import codecs
from os.path import abspath, dirname, join
import numpy as np
from utils.cache import LMDBClient

def load_test_names():
    return data_utils.load_json(settings.DATA_DIR, 'test_name_list2.json')

def load_train_names():
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    return name_to_pubs_train

def testHAN():
    LMDB_NAME_EMB = "lc_attention_network_embedding2"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    han = HAN(lc_emb)

    name_to_pubs_train = load_train_names()
    for name in name_to_pubs_train:
        prec, rec, f1, pids, attentionEmbeddings = han.prepare_and_train(name=name, ispretrain=True, needtSNE=False)
        for pid, attentionEmbedding in zip(pids, attentionEmbeddings):
            lc_emb.set(pid, attentionEmbedding)
        print (name, prec, rec, f1)

def testUser(name):
    LMDB_NAME_EMB = "lc_attention_network_embedding2"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    han = HAN(lc_emb)

    prec, rec, f1, pids, attentionEmbeddings = han.prepare_and_train(name=name, ispretrain=True, needtSNE=False)
    print (name, prec, rec, f1)

if __name__ == '__main__':
    # testHAN()
    name = "gang_yin"
    testUser(name)





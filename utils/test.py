
import lmdb

import codecs
import json

import pickle
from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '..')

DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
PIC_DIR = join(PROJ_DIR, 'pic')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global_')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)


map_size = 1099511627776

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)



def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)

class LMDBClient(object):

    def __init__(self, name, readonly=False):
        try:
            lmdb_dir = join(DATA_DIR, 'lmdb')
            os.makedirs(lmdb_dir, exist_ok=True)
            self.db = lmdb.open(join(lmdb_dir, name), map_size=map_size, readonly=readonly)
        except Exception as e:
            print(e)

    def get(self, key):
        with self.db.begin() as txn:
            value = txn.get(key.encode())
        if value:
            return deserialize_embedding(value)
        else:
            return None

    def getAllDataLength(self):
        with self.db.begin() as txn:
            length = txn.stat()['entries']
            return length

    def get_batch(self, keys):
        values = []
        with self.db.begin() as txn:
            for key in keys:
                value = txn.get(key.encode())
                if value:
                    values.append(deserialize_embedding(value))
        return values

    def set(self, key, vector):
        with self.db.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), serialize_embedding(vector))

    def set_batch(self, generator):
        with self.db.begin(write=True) as txn:
            for key, vector in generator:
                txn.put(key.encode("utf-8"), serialize_embedding(vector))
                print(key, self.get(key))


if __name__ == '__main__':
    LMDB_NAME = "lc_attention_network_embedding2"
    lc_input = LMDBClient(LMDB_NAME)
    name_to_pubs_test = load_json(GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    name = "kexin_xu"
    cur_ret = name_to_pubs_test[name]
    for aid in cur_ret:
        for pid in cur_ret[aid]:
            print (lc_input.get(pid))






from keras.models import  Model
from keras.layers import Input, Dense, Dropout
from utils import data_utils, settings
from utils.cache import LMDBClient
from sklearn.model_selection import train_test_split
import numpy as np

input_dim = 100

input = Input(shape=(input_dim,))
d1 = Dense(50, activation='tanh')(input)
dr1 = Dropout(0.25)(d1)
d2 = Dense(25, activation='tanh')(dr1)
dr2 = Dropout(0.25)(d2)
d3 = Dense(50, activation='tanh')(dr2)
dr3 = Dropout(0.25)(d3)
output = Dense(input_dim, activation='tanh')(dr3)

raw2localTrans = Model(input,output)
raw2localTrans.compile(optimizer='adadelta', loss='binary_crossentropy')

# ==== train ====


rawFeatureLMDBName = "author_100.emb.weighted"
rawFeature = LMDBClient(rawFeatureLMDBName)

LMDB_NAME_EMB = "lc_attention_network_embedding2"
lc_emb = LMDBClient(LMDB_NAME_EMB)


def getPids():
    name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # for test
    cntpapers = []
    for name in name2pubs_train:
        papers = name2pubs_train[name]
        for aid in papers:
            if len(papers[aid]) < 5:
                continue
            for pid in papers[aid]:
                cntpapers.append(pid)
    return cntpapers

def getRawEmbedding(pids):
    rawEmbedding = []
    for pid in pids:
        rawEmbedding.append(rawFeature.get(pid))
    rawEmbedding = np.array(rawEmbedding)
    rawEmbedding.reshape(-1,1)
    return rawEmbedding

def getlocalTransEmbedding(pids):
    TransEmbedding = []
    for pid in pids:
        emb = lc_emb.get(pid)
        TransEmbedding.append(list(emb))
    TransEmbedding = np.array(TransEmbedding)
    TransEmbedding.reshape(-1,1)
    return TransEmbedding


pids = getPids()
rawEmbedding = getRawEmbedding(pids)
TransEmbedding = getlocalTransEmbedding(pids)

X_train, X_test, y_train, y_test = train_test_split(rawEmbedding, TransEmbedding, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

raw2localTrans.fit(X_train, y_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_val, y_val))


res = raw2localTrans.evaluate(X_test, y_test)

# ============= get test data new embedding ============

def load_test_names():
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    return name_to_pubs_test

TestDataPids = []
name_to_test = load_test_names()
for name in name_to_test:
    for aid in name_to_test[name]:
        if len(name_to_test[name][aid]) < 5:
            for pid in name_to_test[name][aid]:
                TestDataPids.append(pid)


def getRawEmbedding(pids):
    rawEmbedding = []
    for pid in pids:
        rawEmbedding.append(rawFeature.get(pid))
    rawEmbedding = np.array(rawEmbedding)
    rawEmbedding.reshape(-1,1)
    return rawEmbedding

TestDataEmbedding = getRawEmbedding(TestDataPids)
transformEmbedding = raw2localTrans.predict(TestDataEmbedding)

LMDB_NAME_EMB = "raw_transform_local_embedding"
lc_emb = LMDBClient(LMDB_NAME_EMB)

# print (transformEmbedding)

# for pid, embedd in zip(TestDataEmbedding, transformEmbedding):
    # print (pid, embedd)
    # lc_emb.set(pid, np.array(embedd))

print ("done")

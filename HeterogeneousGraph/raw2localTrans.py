

from keras.models import  Model
from keras.layers import Input, Dense, Dropout
from utils import data_utils, settings
from utils.cache import LMDBClient
from sklearn.model_selection import train_test_split
import numpy as np

input_dim = 100

input = Input(shape=(input_dim,))
d1 = Dense(50)(input)
dr1 = Dropout(0.25)(d1)
d2 = Dense(25)(dr1)
dr2 = Dropout(0.25)(d2)
d3 = Dense(50)(dr2)
dr3 = Dropout(0.25)(d3)
output = Dense(input_dim)(dr3)

raw2localTrans = Model(input,output)
raw2localTrans.compile(optimizer='adadelta', loss='binary_crossentropy')

# ==== train ====


rawFeatureLMDBName = "author_100.emb.weighted"
rawFeature = LMDBClient(rawFeatureLMDBName)

LMDB_NAME_EMB = "lc_attention_network_embedding2"
lc_emb = LMDBClient(LMDB_NAME_EMB)


def getPids():
    pids = []
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    for name in name_to_pubs_train:
        for aid in name_to_pubs_train[name]:
            for pid in name_to_pubs_train[name][aid]:
                pids.append(pid)
    return pids

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
        TransEmbedding.append(lc_emb.get(pid))
    TransEmbedding = np.array(TransEmbedding)
    TransEmbedding.reshape(-1,1)
    return TransEmbedding


pids = getPids()
rawEmbedding = getRawEmbedding(pids)
TransEmbedding = getlocalTransEmbedding(pids)

# X_train, X_test, y_train, y_test = train_test_split(rawEmbedding, TransEmbedding, test_size=0.2, random_state=1)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

print (rawEmbedding.shape)
print (TransEmbedding.shape)

print (TransEmbedding)

# raw2localTrans.fit(X_train, y_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(X_val, y_val))





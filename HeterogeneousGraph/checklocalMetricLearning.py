
from utils.cache import LMDBClient, data_utils,settings

LMDB_NAME_EMB = "lc_attention_network_embedding"
lc_emb = LMDBClient(LMDB_NAME_EMB)

print (lc_emb.getAllDataLength())

name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # for test
cnt = 0
for name in name2pubs_train:
    papers = name2pubs_train[name]
    for aid in papers:
        for pid in papers[aid]:
            cnt += 1

print ("all number of paper: ", cnt)
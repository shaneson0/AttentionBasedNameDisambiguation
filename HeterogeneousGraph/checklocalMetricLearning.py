
from utils.cache import LMDBClient

LMDB_NAME_EMB = "lc_attention_network_embedding"
lc_emb = LMDBClient(LMDB_NAME_EMB)

print (lc_emb.getAllDataLength())

from HeterogeneousGraph.HAN import HAN
from utils.cache import LMDBClient, data_utils,settings

LMDB_NAME_EMB = "lc_attention_network_embedding2"
lc_emb = LMDBClient(LMDB_NAME_EMB)

NoneCnt = 0
name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # for test
cnt = 0
cntpapers = []
for name in name2pubs_train:
    papers = name2pubs_train[name]
    for aid in papers:
        if len(papers[aid]) < 5:
            continue
        for pid in papers[aid]:
            cntpapers.append(pid)
            emb = lc_emb.get(pid)
            if emb is None:
                NoneCnt += 1

print ("None Cnt: ", NoneCnt)




# print ("all number of paper: ", len(set(cntpapers)))
#
#
# def testHAN():
#     han = HAN(lc_emb)
#     cntpapers = []
#
#     for name in name2pubs_train:
#         rawFeatures, labels, pids, rawlabels = han.loadFeature(name, ispretrain=True)
#         for pid in pids:
#             cntpapers.append(pid)
#
#     return len(set(cntpapers))
#
# print ("all number of train pids: ", testHAN())



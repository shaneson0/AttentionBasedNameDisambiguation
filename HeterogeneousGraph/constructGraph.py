from os.path import join
import os
import numpy as np
from numpy.random import shuffle
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils, inputData
from utils import settings, string_utils
from collections import defaultdict
from HeterogeneousGraph import IDF_THRESHOLD, Author_THRESHOLD

IDLength = 24


pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')
Author2Id = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'Author2Id.json')


def getLabelId(pid, authorName):
    authorName = string_utils.clean_name(authorName)
    PapaerInfo = pubs_dict[pid]
    authors = PapaerInfo['authors']
    # print(authorName, authors)
    for author in authors:
        if string_utils.clean_name(author['name']) == authorName:
            print('get the same')
            return Author2Id[author['name'] + ':' + author.get('org', 'null')]
    return -1

def  genPAPandPSP(idf_threshold=10):
    AuthorSocial = inputData.loadAuthorSocial()

    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
    raw_word2vec = 'author_100.emb.weighted'
    lc_emb = LMDBClient(raw_word2vec)

    LMDB_AUTHOR_FEATURE = "pub_authors.feature"
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        print(i, name)
        cur_person_dict = name_to_pubs_test[name]
        pids_set = set()
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_feature_and_label.txt'.format(name)), 'w')
        for i, aid in enumerate(cur_person_dict):
            items = cur_person_dict[aid]
            if len(items) < 5:
                continue
            for pid in items:
                pids2label[pid] = aid
                pids.append(pid)
        shuffle(pids)

        for pid in pids:
            # use raw feature rather than Triplet Loss
            cur_pub_emb = lc_emb.get(pid)
            # cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}'.format(pids2label[pid]))
                LabelId = getLabelId(pid[:IDLength], name)
                wf_content.write('\t{}\n'.format(LabelId))
        wf_content.close()

        # generate network1
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_PAP.txt'.format(name)), 'w')
        for i in range(n_pubs-1):
            if i % 10 == 0:
                print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i+1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                # print('author_feature2: ', author_feature2)
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))
        wf_network.close()


        def CountNumber(A, B):
            res = 0
            for x in A:
                for y in B:
                    if x == y:
                        res = res + 1

            return res

        wf_network = open(join(graph_dir, '{}_PSP.txt'.format(name)), 'w')

        for i in range(n_pubs-1):
            for j in range(i + 1, n_pubs):
                Graph1Socials = AuthorSocial[pids_filter[i]]
                Graph2Socials = AuthorSocial[pids_filter[j]]
                if CountNumber(Graph1Socials, Graph2Socials) >= Author_THRESHOLD:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))

        wf_network.close()




if __name__ == '__main__':
    # test_prepare_local_data('hongbin_li')
    genPAPandPSP(idf_threshold=IDF_THRESHOLD)
    print('done')







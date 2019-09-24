# coding=utf-8

from os.path import abspath, dirname, join
import sys

PROJ_DIR = join(abspath(dirname(__file__)), '..')
sys.path.append(PROJ_DIR)

from collections import Counter
from utils import settings
from os.path import join
import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import time
from model.preprocessing import gen_train_edges, preprocess_graph, normalize_vectors, sparse_to_tuple, construct_feed_dict
from sklearn.manifold import TSNE
from finch import FINCH
from os.path import abspath, dirname, join
from utils import getSetting, PCAAnanlyse, clustering, pairwise_precision_recall_f1, lossPrint, tSNEAnanlyse, settings, sNEComparingAnanlyse
from utils.inputData import load_local_data
from sklearn.metrics import silhouette_score

from model import DualGCNGraphFusion, OptimizerDualGCNAutoEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


flags = getSetting()
FLAGS = flags.FLAGS
model_str = FLAGS.model
name_str = FLAGS.name

def AdjPreprocessing(adj):
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)
    return adj_train

def BuildModel(placeholders, input_feature_dim, num_nodes, name, num_logits):
    Model = DualGCNGraphFusion(placeholders, input_feature_dim, num_nodes, name=name, num_logits=num_logits)
    return Model

# def BuildOptimizer()
def NormalizedAdj(adj):
    adj_train = AdjPreprocessing(adj)
    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    return adj_norm, adj_label

def getOriginClusterLabel(originClusterlabels, CurrentClusterLabels, idx):
    OriginLabel = originClusterlabels[idx]
    RelationNewLabels = []
    for i, label in enumerate(originClusterlabels):
        if label == OriginLabel and idx != i:
            RelationNewLabels.append(CurrentClusterLabels[i])
    if len(RelationNewLabels) == 0:
        return -1
    print ('RelationNewLabels: ', RelationNewLabels)
    a = np.array(RelationNewLabels)
    counts = np.bincount(a)
    return np.argmax(counts)

def toOneHot(Clusterlabels):
    le = LabelEncoder()
    le_clusterlabel = le.fit(Clusterlabels)
    clusterLabel = le_clusterlabel.transform(Clusterlabels)
    ohe_clusterlabel = OneHotEncoder(sparse=False).fit(clusterLabel.reshape(-1, 1))
    Sex_ohe = ohe_clusterlabel.transform(clusterLabel.reshape(-1, 1))

    return Sex_ohe

def getNewClusterLabel(emb, initClusterlabel, NumberOfCluster):
    Clusterlabels = clustering(emb, num_clusters=NumberOfCluster)

    print ('Clusterlabels: ', Counter(Clusterlabels))
    print ('initClusterlabel: ', initClusterlabel)
    # 假如出现只有一种类别的话，这个要做修改和调整的。
    C = Counter(Clusterlabels)
    # print (C)
    for idx, v in C.items():
        if v == 1:
            tTable = getOriginClusterLabel(initClusterlabel, Clusterlabels, idx)
            if tTable == -1:
                continue
            print ('idx: ', idx, ', tTable: ', tTable)
            for tidx, k in enumerate(Clusterlabels):
                if Clusterlabels[tidx] == idx:
                    Clusterlabels[tidx] = tTable

            # 删了一个label，后面的label往前移
            for tidx, k in enumerate(Clusterlabels):
                if Clusterlabels[tidx] > idx:
                    Clusterlabels[tidx] = Clusterlabels[tidx] - 1
            NumberOfCluster = NumberOfCluster - 1
    # Clusterlabels = clustering(emb, num_clusters=NumberOfCluster)

    return NumberOfCluster, Clusterlabels

def train(name, needtSNE=False, savefile=True):
    adj, adj2, features, labels, Clusterlabels = load_local_data(name=name)


    initClusterlabel = Clusterlabels
    oneHotClusterLabels = toOneHot(Clusterlabels)
    num_logits = len(oneHotClusterLabels[0])
    # enc.transform([['Female', 1], ['Male', 4]]).toarray()
    print ('debuging ', oneHotClusterLabels.shape)

    originClusterlabels = Clusterlabels
    n_clusters = len(set(labels))
    OldClusterlabels = Clusterlabels
    originNumberOfClusterlabels = len(set(Clusterlabels))

    num_nodes = adj.shape[0]
    input_feature_dim = features.shape[1]
    adj_norm, adj_label = NormalizedAdj(adj)
    adj_norm2, adj_label2 = NormalizedAdj(adj2)


    if FLAGS.is_sparse:  # TODO to test
        # features = sparse_to_tuple(features.tocoo())
        # features_nonzero = features[1].shape[0]
        features = features.todense()  # TODO
    else:
        features = normalize_vectors(features)

    # Define placeholders
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, input_feature_dim)),
        'labels': tf.placeholder(tf.int64, shape=(None), name='labels'),
        'graph1': tf.sparse_placeholder(tf.float32),
        'graph2': tf.sparse_placeholder(tf.float32),
        'graph1_orig': tf.sparse_placeholder(tf.float32),
        'graph2_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'epoch': tf.placeholder_with_default(0., shape=()),
        'clusterEpoch': tf.placeholder_with_default(0., shape=())
    }
    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_3_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    def getGraphDetail(adj):
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)
        return {'norm': norm, 'pos_weight': pos_weight}

        # return pos_weight, norm
    # loss1s = []
    # loss2s = []
    # loss3s = []

    n_clusters = len(set(labels))
    graph1 = getGraphDetail(adj)
    graph2 = getGraphDetail(adj2)

    # construct adj_orig
    graph1['labels'] = tf.reshape(tf.sparse_tensor_to_dense(placeholders['graph1_orig'],
                                                                           validate_indices=False), [-1])
    graph2['labels'] = tf.reshape(tf.sparse_tensor_to_dense(placeholders['graph2_orig'],
                                                                           validate_indices=False), [-1])

    # Train model
    for clusterepoch in range(FLAGS.clusterEpochs):
        print ('cluster epoch: ', clusterepoch)
        # tf.reset_default_graph()


        # num_logits
        model = BuildModel(placeholders, input_feature_dim, num_nodes, name='model%d'%(clusterepoch), num_logits=num_logits)

        # Session

        # tf.reset_default_graph()
        # sess = tf.InteractiveSession()

        opt = OptimizerDualGCNAutoEncoder(model=model,
                                          num_nodes=num_nodes,
                                          z_label=Clusterlabels,
                                          name='model%d' % (clusterepoch),
                                          graph1=graph1,
                                          graph2=graph2
                                          )

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Centers
        # centers = opt.centers

        for epoch in range(FLAGS.epochs):

            model.epoch = epoch

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, adj_norm2, adj_label2, features, placeholders, Clusterlabels, epoch, clusterepoch+1)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            # outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
            outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

            [cost, reconstructloss, reconstructloss1, reconstructloss2,kl, centerloss] = sess.run([opt.cost, opt.reconstructloss, opt.reconstructloss1, opt.reconstructloss2, opt.kl, opt.centerloss], feed_dict=feed_dict)

            print ('epoch: ', epoch, '， cost: ', cost, ', reconstructloss: ', reconstructloss, ', reconstructloss1: ', reconstructloss1, ', reconstructloss2 : ', reconstructloss2, ',kl : ', kl, ', centerloss: ', centerloss)

        # if clusterepoch != FLAGS.clusterEpochs -1 :
        emb = get_embs()
        X_new = TSNE(learning_rate=100).fit_transform(emb)

        tClusterLabels = []
        Maxscore = -10000
        NumberOfCluster = 0
        for nc in range(2, originNumberOfClusterlabels+1, 1):
            TempLabels = clustering(X_new, nc)
            score = silhouette_score(X_new, TempLabels)
            print ('nc: ', nc, ', score: ', score)
            if score > Maxscore:
                Maxscore = score
                tClusterLabels = TempLabels
                NumberOfCluster = nc

        print ('NumberOfCluster: ', NumberOfCluster, ', originNumberOfClusterlabels : ', originNumberOfClusterlabels, ', Maxscore: ', Maxscore)
        if NumberOfCluster < 0 or NumberOfCluster > originNumberOfClusterlabels:
            continue

        # 符合不断缩小的要求
        # 重新修改这些参数
        Clusterlabels = tClusterLabels
        originNumberOfClusterlabels = NumberOfCluster

        prec, rec, f1 = pairwise_precision_recall_f1(Clusterlabels, labels)
        print ('prec: ', prec, ', rec: ', rec, ', f1: ', f1, ', originNumberOfClusterlabels: ', originNumberOfClusterlabels)
        Cc = Counter(Clusterlabels)
        print (Cc)
        if needtSNE:
            sNEComparingAnanlyse(emb, OldClusterlabels, labels, Clusterlabels, savepath= join(settings.PIC_DIR,  "%s_%s.png"%(name, clusterepoch)) )
            # tSNEAnanlyse(emb, labels, join(settings.PIC_DIR, "%s.png"%(clusterepoch)) )
            # tf.reset_default_graph()

    emb = get_embs()
    emb_norm = normalize_vectors(emb)
    clusters_pred = clustering(emb_norm, num_clusters=originNumberOfClusterlabels)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    print ('prec: ', prec, ', rec: ', rec, ', f1: ', f1, ', originNumberOfClusterlabels: ', originNumberOfClusterlabels)
    # lossPrint(range(FLAGS.epochs), loss1s, loss2s, loss3s)
    if needtSNE:
        tSNEAnanlyse(emb, labels, join(settings.PIC_DIR,  "%s_final.png"%(name)) )
    tf.reset_default_graph()
    return [prec, rec, f1], num_nodes, n_clusters

def test(name):
    FLAGS.DGAE_learning_rate = 0.03
    print(FLAGS.DGAE_learning_rate)
    # adj, adj2, features, labels, Clusterlabels = load_local_data(name=name)
    # print(features)
    # print(labels)
    # print(Clusterlabels)

from utils import data_utils, eval_utils
def load_test_names():
    return data_utils.load_json(settings.DATA_DIR, 'test_name_list2.json')

def main():
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')
    metrics = np.zeros(3)
    cnt = 0
    for name in names:
        print('name : ', name)
        cur_metric, num_nodes, n_clusters = train(name, True)
        wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}\n'.format(
            name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2]))
        wf.flush()
        for i, m in enumerate(cur_metric):
            metrics[i] += m
        cnt += 1
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_f1 = eval_utils.cal_f1(macro_prec, macro_rec)
        print('average until now', [macro_prec, macro_rec, macro_f1])
        # time_acc = time.time()-start_time
        # print(cnt, 'names', time_acc, 'avg time', time_acc/cnt)
    macro_prec = metrics[0] / cnt
    macro_rec = metrics[1] / cnt
    macro_f1 = eval_utils.cal_f1(macro_prec, macro_rec)
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1))
    wf.close()

if __name__ == '__main__':
    # main()
    # train('kexin_xu', needtSNE=True, savefile=True)
    # test('kexin_xu')
    train('hongbin_li', needtSNE=True, savefile=True)
    # test('hongbin_li')



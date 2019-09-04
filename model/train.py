from os.path import join
import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import time
from model.preprocessing import gen_train_edges, preprocess_graph, normalize_vectors, sparse_to_tuple, construct_feed_dict
from sklearn.manifold import TSNE

from utils import getSetting, PCAAnanlyse, clustering, pairwise_precision_recall_f1, lossPrint, tSNEAnanlyse, settings
from utils.inputData import load_local_data

from model import DualGCNGraphFusion, OptimizerDualGCNAutoEncoder



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

def BuildModel(placeholders, input_feature_dim, num_nodes):
    Model = DualGCNGraphFusion(placeholders, input_feature_dim, num_nodes)
    return Model

# def BuildOptimizer()

def NormalizedAdj(adj):
    adj_train = AdjPreprocessing(adj)
    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    return adj_norm, adj_label

def train(name, needtSNE=False):
    adj, adj2, features, labels, Clusterlabels = load_local_data(name=name)

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
        'graph1': tf.sparse_placeholder(tf.float32),
        'graph2': tf.sparse_placeholder(tf.float32),
        'graph1_orig': tf.sparse_placeholder(tf.float32),
        'graph2_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    model = BuildModel(placeholders, input_feature_dim, num_nodes)
    opt = OptimizerDualGCNAutoEncoder(preds_1=model.reconstructions_1,
                                      labels_1=tf.reshape(tf.sparse_tensor_to_dense(placeholders['graph1_orig'],validate_indices=False), [-1]),
                                      preds_2=model.reconstructions_1,
                                      labels_2=tf.reshape(tf.sparse_tensor_to_dense(placeholders['graph2_orig'],validate_indices=False), [-1]),
                                      model=model,
                                      num_nodes=num_nodes,
                                      pos_weight=pos_weight,
                                      norm=norm,
                                      z_label=Clusterlabels
                                      )

    # Centers
    centers = opt.centers

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_3, feed_dict=feed_dict)  # z_mean is better
        return emb

    loss1s = []
    loss2s = []
    loss3s = []
    # Train model
    for epoch in range(FLAGS.epochs):

        # opt.epoch = epoch
        model.epoch = epoch

        t = time.time()
        # Construct feed dictionary

        feed_dict = construct_feed_dict(adj_norm, adj_label, adj_norm2, adj_label2, features, placeholders, Clusterlabels)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        Loss = sess.run(opt.cost, feed_dict=feed_dict)
        loss1 = sess.run(opt.loss1, feed_dict=feed_dict)
        loss2 = sess.run(opt.loss2, feed_dict=feed_dict)
        # loss3 = sess.run(opt.loss3, feed_dict=feed_dict)
        centerloss = sess.run(opt.centerloss, feed_dict=feed_dict)

        print ('loss: ', Loss, ', loss1: ', loss1, ', loss2: ', loss2 ,', centerloss: ', centerloss, ', acc: ', outs[2])
        # print ('loss: ', Loss, ', loss1: ', loss1, ', loss2: ', loss2, ', loss3: ', loss3, ', centerloss: ', centerloss, ', acc: ', outs[2])
        loss1s.append(loss1)
        loss2s.append(loss2)

        n_clusters = len(set(labels))

        if epoch != 0 and epoch % 1000 == 0:
            Centers = sess.run(centers)
            X_new = TSNE(learning_rate=100).fit_transform(Centers)

            print('centers: ', Centers)

            # 求出最大的distance 和 最小的distance
            maxdistance = -1
            mindistance = 100
            Len = len(X_new)
            ChangeLabels = {i:i for i in range(Len)}
            distance = []
            for i in range(Len):
                for j in range(i+1, Len, 1):
                    t = np.linalg.norm(X_new[i] - X_new[j])
                    if maxdistance < t:
                        maxdistance = t
                    if mindistance > t:
                        mindistance = t
                    if t < 1:
                        # j => i
                        ChangeLabels[j] = ChangeLabels[i]
                    distance = t

            for k in range(len(Clusterlabels)):
                if Clusterlabels[k] in ChangeLabels:
                    Clusterlabels[k] = ChangeLabels[ Clusterlabels[k] ]

            # opt.updateLabel(Clusterlabels)
            opt.updateVariable(epoch)

            print('distance :', distance)
            print('maxdistance: ', maxdistance, ', mindistance: ', mindistance)
            # print()
            # TempClusterlabels = sess.run(Clusterlabels)
            print(Clusterlabels)
            print('len(list(set(Clusterlabels)))', len(list(set(Clusterlabels))))
            # print(len(Centers))

            emb = get_embs()
            tSNEAnanlyse(emb, labels)

        # loss3s.append(loss3)


    emb = get_embs()

    emb_norm = normalize_vectors(emb)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    print ('prec: ', prec, ', rec: ', rec, ', f1: ', f1)
    # lossPrint(range(FLAGS.epochs), loss1s, loss2s, loss3s)
    if needtSNE:
        tSNEAnanlyse(emb, labels)
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
        cur_metric, num_nodes, n_clusters = train(name)
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
    train('kexin_xu', needtSNE=True)
    # test('kexin_xu')
    # train('hongbin_li')
    # test('hongbin_li')



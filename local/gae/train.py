from __future__ import division
from __future__ import print_function

import os
import time
from os.path import join
from utils import settings
# from global_.prepare_local_data import IDF_THRESHOLD

IDF_THRESHOLD = 32
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESHOLD))


import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from local.gae.optimizer import OptimizerAE, OptimizerVAE
from local.gae.model import GCNModelAE, GCNModelVAE
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from utils.cluster import clustering
from utils.data_utils import load_json
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils import settings, tSNEAnanlyse


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()

from utils.cache import LMDBClient
INTER_LMDB_NAME = 'triplete_loss_lc_attention_network_embedding'
lc_inter = LMDBClient(INTER_LMDB_NAME)

RAW_INTER_NAME = 'author_100.emb.weighted'
lc_inter_raw = LMDBClient(INTER_LMDB_NAME)



def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))

def load_local_data(path=local_na_dir, name='cheng_cheng', rawfeature=False):
    # Load local paper network dataset
    print('Loading {} dataset...'.format(name), 'path=', path, name)

    idx_features_labels = np.genfromtxt(join(path , "{}_pubs_content.txt".format(name)), dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?
    labels = encode_labels(idx_features_labels[:, -2])

    features = []
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    for id in idx:
        if rawfeature is False:
            features.append(lc_inter.get(id))
        else:
            features.append(lc_inter_raw.get(id))
    features = np.array(features, dtype=np.float32)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(join(path, "{}_pubs_network.txt".format(name)), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return adj, features, labels

def gae_for_na(name):
    """
    train and evaluate disambiguation results for a specific name
    :param name:  author name
    :return: evaluation results
    """
    adj, features, labels = load_local_data(name=name)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    input_feature_dim = features.shape[1]
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
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, input_feature_dim)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    print('positive edge weight', pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "time=", "{:.5f}".format(time.time() - t))

    emb = get_embs()
    n_clusters = len(set(labels))
    emb_norm = normalize_vectors(emb)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))

    clusters_pred2 = clustering(features, num_clusters=n_clusters)
    prec2, rec2, f12 =  pairwise_precision_recall_f1(clusters_pred2, labels)
    print('pairwise precision', '{:.5f}'.format(prec2),
          'recall', '{:.5f}'.format(rec2),
          'f1', '{:.5f}'.format(f12))

    tSNEAnanlyse(emb_norm, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_final.png" % (name)))
    tSNEAnanlyse(features, labels, join(settings.PIC_DIR, "FINALResult", "rawReature_%s_gae_features.png" % (name)))


    return [prec, rec, f1], num_nodes, n_clusters


def load_test_names():
    return load_json(settings.DATA_DIR, 'test_name_list2.json')


def main():
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')
    metrics = np.zeros(3)
    cnt = 0
    for name in names:
        cur_metric, num_nodes, n_clusters = gae_for_na(name)
        wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}\n'.format(
            name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2]))
        wf.flush()
        for i, m in enumerate(cur_metric):
            metrics[i] += m
        cnt += 1
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_f1 = cal_f1(macro_prec, macro_rec)
        print('average until now', [macro_prec, macro_rec, macro_f1])
        time_acc = time.time()-start_time
        print(cnt, 'names', time_acc, 'avg time', time_acc/cnt)
    macro_prec = metrics[0] / cnt
    macro_rec = metrics[1] / cnt
    macro_f1 = cal_f1(macro_prec, macro_rec)
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1))
    wf.close()




if __name__ == '__main__':
    # gae_for_na('hongbin_li')
    # gae_for_na('j_yu')
    gae_for_na('kexin_xu')
    # main()
# 650 hongbin_li_pubs_network.txt
# 9459 hongbin_li_pubs_network.txt

# 12987 kexin_xu_pubs_network.txt
# 1354 data/local/graph-32/kexin_xu_pubs_network.txt
# 2671 kexin_xu_pubs_network.txt
# 5712 data/local/graph-32/kexin_xu_pubs_network.txt




import codecs
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import time
from model.preprocessing import gen_train_edges, preprocess_graph, normalize_vectors, sparse_to_tuple, construct_feed_dict

from utils import getSetting, PCAAnanlyse, clustering, pairwise_precision_recall_f1, lossPrint
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

def train(name):
    adj, adj2, features, labels = load_local_data(name=name)

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
                                      norm=norm
                                      )
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

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, adj_norm2, adj_label2, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        Loss = sess.run(opt.cost, feed_dict=feed_dict)
        loss1 = sess.run(opt.loss1, feed_dict=feed_dict)
        loss2 = sess.run(opt.loss2, feed_dict=feed_dict)
        loss3 = sess.run(opt.loss3, feed_dict=feed_dict)

        print ('loss: ', Loss, ', loss1: ', loss1, ', loss2: ', loss2, ', loss3: ', loss3, ', acc: ', outs[2])
        loss1s.append(loss1)
        loss2s.append(loss2)
        loss3s.append(loss3)


    emb = get_embs()

    n_clusters = len(set(labels))
    emb_norm = normalize_vectors(emb)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
    print ('prec: ', prec, ', rec: ', rec, ', f1: ', f1)
    lossPrint(range(FLAGS.epochs), loss1s, loss2s, loss3s)
    PCAAnanlyse(emb, labels)



if __name__ == '__main__':
    train('hongbin_li')
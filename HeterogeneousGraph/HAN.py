
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
from utils import my_KNN, my_Kmeans  # , my_TSNE, my_Linear
from models import GAT, HeteGAT, HeteGAT_multi
from utils import process
from os.path import join
from utils import settings, string_utils
# 禁用gpu
import os
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'acm'
featype = 'fea'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 200
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

IDF_THRESHOLD = 32  # small data



class HAN():
    def __init__(self):
        self.enc = OneHotEncoder()

    def encode_labels(self, labels):
        classes = set(labels)
        classes_dict = {c: i for i, c in enumerate(classes)}
        res = [[label, classes_dict[label]] for label in labels]
        return self.enc.fit_transform(res).toarray()

    def constructAdj(self, pids):
        pid2idx = {c: i for i, c in enumerate(pids)}
        idx2pid = {i: c for i, c in enumerate(pids)}

        LenPids = len(pids)
        PAP = np.zeros(shape=(LenPids, LenPids))
        PSP = np.zeros(shape=(LenPids, LenPids))
        return PAP, PSP, pid2idx, idx2pid

    def constructIdx(self, X):
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=1)
        return X_train, X_val, X_test

    def getPATH(self, name, idf_threshold, filename):
        graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
        path = join(graph_dir, '{}_{}.txt'.format(name, filename))
        return path

    def loadFeature(self, name, idf_threshold=IDF_THRESHOLD):
        featurePath = self.getPATH(name, idf_threshold, 'feature_and_label')
        # idx_features_labels = np.genfromtxt(join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)), dtype=np.dtype(str))
        idx_features_labels = np.genfromtxt(featurePath, dtype=np.dtype(str))
        features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?
        labels = self.encode_labels(idx_features_labels[:, -2])
        pids = idx_features_labels[:, 0]
        return features, labels, pids

    def loadPAP(self, PAP, pid2idx, name, idf_threshold=IDF_THRESHOLD):
        PAPPATH = self.getPATH(name, idf_threshold, 'PAP')
        PAPPath = np.genfromtxt(PAPPATH, dtype=np.dtype(str))
        for _from, _to in PAPPath:
            PAP[pid2idx[_from]][pid2idx[_to]] = 1
            PAP[pid2idx[_to]][pid2idx[_from]] = 1
        return PAP

    def loadPSP(self, PSP, pid2idx, name, idf_threshold=IDF_THRESHOLD):
        PSPPATH = self.getPATH(name, idf_threshold, 'PSP')
        PSPPath = np.genfromtxt(PSPPATH, dtype=np.dtype(str))
        for _from, _to in PSPPath:
            PSP[pid2idx[_from]][pid2idx[_to]] = 1
            PSP[pid2idx[_to]][pid2idx[_from]] = 1
        return PSP

    def load_data_dblp(self, truelabels, truefeatures, PAP, PSP, train_idx, val_idx, test_idx):
        rownetworks = [PAP, PSP]

        y = truelabels

        train_mask = self.sample_mask(train_idx, y.shape[0])
        val_mask = self.sample_mask(val_idx, y.shape[0])
        test_mask = self.sample_mask(test_idx, y.shape[0])

        y_train = np.zeros(y.shape)
        y_val = np.zeros(y.shape)
        y_test = np.zeros(y.shape)
        y_train[train_mask, :] = y[train_mask, :]
        y_val[val_mask, :] = y[val_mask, :]
        y_test[test_mask, :] = y[test_mask, :]

        # return selected_idx, selected_idx_2
        print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                              y_val.shape,
                                                                                              y_test.shape,
                                                                                              train_idx.shape,
                                                                                              val_idx.shape,
                                                                                              test_idx.shape))
        truefeatures_list = [truefeatures, truefeatures, truefeatures]
        return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask



    def prepare_and_train(self):

        name = 'zhigang_zeng'
        # loadData(name)
        features, labels, pids = self.loadFeature(name)
        PAP, PSP, pid2idx, idx2pid = self.constructAdj(pids)

        PAP = self.loadPAP(PAP, pid2idx, name)
        print (PAP)
        print (PAP.tolist())

        PSP = self.loadPSP(PSP, pid2idx, name)
        print (PSP)
        print (PSP.tolist())

        N = len(pids)
        X_train, X_val, X_test = self.constructIdx(list(range(N)))
        print (X_train, X_val, X_test)


        adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = self.load_data_dblp(labels,
                                                                                                         features, PAP,
                                                                                                         PSP, X_train,
                                                                                                         X_val, X_test)
        self.train(adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask)



    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def load_data_dblp(self, truelabels, truefeatures, PAP, PSP, train_idx, val_idx, test_idx):
        rownetworks = [PAP, PSP]

        y = truelabels

        train_mask = self.sample_mask(train_idx, y.shape[0])
        val_mask = self.sample_mask(val_idx, y.shape[0])
        test_mask = self.sample_mask(test_idx, y.shape[0])

        y_train = np.zeros(y.shape)
        y_val = np.zeros(y.shape)
        y_test = np.zeros(y.shape)
        y_train[train_mask, :] = y[train_mask, :]
        y_val[val_mask, :] = y[val_mask, :]
        y_test[test_mask, :] = y[test_mask, :]

        # # return selected_idx, selected_idx_2
        # print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
        #                                                                                       y_val.shape,
        #                                                                                       y_test.shape,
        #                                                                                       train_idx.shape,
        #                                                                                       val_idx.shape,
        #                                                                                       test_idx.shape))
        truefeatures_list = [truefeatures, truefeatures, truefeatures]
        return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask

    def getLabel(self, Y):
        Tlabels = self.enc.inverse_transform(Y)
        labels = [T[1] for T in Tlabels]
        return labels, len(set(labels))

    def train(self, adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask):


        nb_nodes = fea_list[0].shape[0]
        ft_size = fea_list[0].shape[1]
        nb_classes = y_train.shape[1]

        # adj = adj.todense()

        # features = features[np.newaxis]  # [1, nb_node, ft_size]
        fea_list = [fea[np.newaxis] for fea in fea_list]
        adj_list = [adj[np.newaxis] for adj in adj_list]
        y_train = y_train[np.newaxis]
        y_val = y_val[np.newaxis]
        y_test = y_test[np.newaxis]
        train_mask = train_mask[np.newaxis]
        val_mask = val_mask[np.newaxis]
        test_mask = test_mask[np.newaxis]

        biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

        print('build graph...')
        with tf.Graph().as_default():
            with tf.name_scope('input'):
                ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                              shape=(batch_size, nb_nodes, ft_size),
                                              name='ftr_in_{}'.format(i))
                               for i in range(len(fea_list))]
                bias_in_list = [tf.placeholder(dtype=tf.float32,
                                               shape=(batch_size, nb_nodes, nb_nodes),
                                               name='bias_in_{}'.format(i))
                                for i in range(len(biases_list))]
                lbl_in = tf.placeholder(dtype=tf.int32, shape=(
                    batch_size, nb_nodes, nb_classes), name='lbl_in')
                msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                        name='msk_in')
                attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
                ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
                is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
            # forward
            logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                               attn_drop, ffd_drop,
                                                               bias_mat_list=bias_in_list,
                                                               hid_units=hid_units, n_heads=n_heads,
                                                               residual=residual, activation=nonlinearity)

            # cal masked_loss
            log_resh = tf.reshape(logits, [-1, nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])
            loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
            # optimzie
            train_op = model.training(loss, lr, l2_coef)

            saver = tf.train.Saver()

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            vlss_mn = np.inf
            vacc_mx = 0.0
            curr_step = 0

            with tf.Session(config=config) as sess:
                sess.run(init_op)

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

                for epoch in range(nb_epochs):
                    tr_step = 0

                    tr_size = fea_list[0].shape[0]
                    # ================   training    ============
                    while tr_step * batch_size < tr_size:
                        fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                               for i, d in zip(ftr_in_list, fea_list)}
                        fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                               for i, d in zip(bias_in_list, biases_list)}
                        fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                               msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                               is_train: True,
                               attn_drop: 0.6,
                               ffd_drop: 0.6}
                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                           feed_dict=fd)
                        train_loss_avg += loss_value_tr
                        train_acc_avg += acc_tr
                        tr_step += 1

                    vl_step = 0
                    vl_size = fea_list[0].shape[0]
                    # =============   val       =================
                    while vl_step * batch_size < vl_size:
                        # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                        fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                               for i, d in zip(ftr_in_list, fea_list)}
                        fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                               for i, d in zip(bias_in_list, biases_list)}
                        fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                               msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                               is_train: False,
                               attn_drop: 0.0,
                               ffd_drop: 0.0}

                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                         feed_dict=fd)
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1
                    # import pdb; pdb.set_trace()
                    print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
                    print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                          (train_loss_avg / tr_step, train_acc_avg / tr_step,
                           val_loss_avg / vl_step, val_acc_avg / vl_step))

                    if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                        if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                            vacc_early_model = val_acc_avg / vl_step
                            vlss_early_model = val_loss_avg / vl_step
                            saver.save(sess, checkpt_file)
                        vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                        vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                        curr_step = 0
                    else:
                        curr_step += 1
                        if curr_step == patience:
                            print('Early stop! Min loss: ', vlss_mn,
                                  ', Max accuracy: ', vacc_mx)
                            print('Early stop model validation loss: ',
                                  vlss_early_model, ', accuracy: ', vacc_early_model)
                            break

                    train_loss_avg = 0
                    train_acc_avg = 0
                    val_loss_avg = 0
                    val_acc_avg = 0

                saver.restore(sess, checkpt_file)
                print('load model from : {}'.format(checkpt_file))
                ts_size = fea_list[0].shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0

                while ts_step * batch_size < ts_size:
                    # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
                    fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                           for i, d in zip(ftr_in_list, fea_list)}
                    fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                           for i, d in zip(bias_in_list, biases_list)}
                    fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                           msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],

                           is_train: False,
                           attn_drop: 0.0,
                           ffd_drop: 0.0}

                    fd = fd1
                    fd.update(fd2)
                    fd.update(fd3)
                    loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                          feed_dict=fd)
                    ts_loss += loss_value_ts
                    ts_acc += acc_ts
                    ts_step += 1

                print('Test loss:', ts_loss / ts_step,
                      '; Test accuracy:', ts_acc / ts_step)

                print('start knn, kmean.....')
                xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]

                from numpy import linalg as LA

                # xx = xx / LA.norm(xx, axis=1)
                yy = y_test[test_mask]


                print ("XX: ", xx)
                print ("YY: ", yy)
                print('xx: {}, yy: {}'.format(xx.shape, yy.shape))

                labels, numberofLabels = self.getLabel(yy)

                from utils import  clustering, pairwise_precision_recall_f1
                # print ('labels: ', labels)
                clusters_pred = clustering(xx, num_clusters=numberofLabels)
                prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
                print ('prec: ', prec, ', rec: ', rec, ', f1: ', f1, ', originNumberOfClusterlabels: ',
                       numberofLabels)

                # my_KNN(xx, yy)
                # my_Kmeans(xx, yy)

                sess.close()

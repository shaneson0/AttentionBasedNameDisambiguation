# coding=utf-8
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
from utils import my_KNN, my_Kmeans  # , my_TSNE, my_Linear
from models import GAT, HeteGAT, HeteGAT_multi, OSM_CAA_Loss
from utils import process
from os.path import join
from utils import settings, string_utils

import os
from sklearn.model_selection import train_test_split
from utils import getSetting, PCAAnanlyse, clustering, pairwise_precision_recall_f1, lossPrint, tSNEAnanlyse, settings, sNEComparingAnanlyse


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'acm'
featype = 'fea'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 1000
patience = 100
lr = 0.01  # learning rate
l2_coef = 0.0001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [10]
n_heads = [10, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

from HeterogeneousGraph import IDF_THRESHOLD



class HAN():
    def __init__(self):
        self.enc = OneHotEncoder()

    def encode_labels(self, labels):
        classes = set(labels)
        classes_dict = {c: i for i, c in enumerate(classes)}
        res = [[label, classes_dict[label]] for label in labels]
        rawlabels = [classes_dict[label] for label in labels]
        return self.enc.fit_transform(res).toarray(), np.array(rawlabels)

    def constructAdj(self, pids):
        pid2idx = {c: i for i, c in enumerate(pids)}
        idx2pid = {i: c for i, c in enumerate(pids)}

        LenPids = len(pids)
        PAP = np.zeros(shape=(LenPids, LenPids))
        PSP = np.zeros(shape=(LenPids, LenPids))
        return PAP, PSP, pid2idx, idx2pid

    def constructIdx(self, X, labels):
        X_train, X_val = train_test_split(X, stratify=labels, test_size=0.2, random_state=1)
        return X_train, X_val, X, X

    def getPATH(self, name, idf_threshold, filename):
        graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
        path = join(graph_dir, '{}_{}.txt'.format(name, filename))
        return path

    def loadFeature(self, name, idf_threshold=IDF_THRESHOLD):
        featurePath = self.getPATH(name, idf_threshold, 'feature_and_label')
        # idx_features_labels = np.genfromtxt(join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold)), dtype=np.dtype(str))
        idx_features_labels = np.genfromtxt(featurePath, dtype=np.dtype(str))
        features = np.array(idx_features_labels[:, 1:-2], dtype=np.float32)  # sparse?
        labels, rawlabels = self.encode_labels(idx_features_labels[:, -2])
        pids = idx_features_labels[:, 0]
        return features, labels, pids, rawlabels

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

    def load_data_dblp(self, truelabels, rawlabels, truefeatures, PAP, PSP, train_idx, val_idx, test_idx, allIdx):
        rownetworks = [PAP, PSP]

        y = truelabels

        all_mask = self.sample_mask(allIdx, y.shape[0])
        train_mask = self.sample_mask(train_idx, y.shape[0])
        val_mask = self.sample_mask(val_idx, y.shape[0])
        test_mask = self.sample_mask(test_idx, y.shape[0])

        y_all = np.zeros(y.shape)
        y_train = np.zeros(y.shape)
        y_val = np.zeros(y.shape)
        y_test = np.zeros(y.shape)
        y_train[train_mask, :] = y[train_mask, :]
        y_val[val_mask, :] = y[val_mask, :]
        y_test[test_mask, :] = y[test_mask, :]
        y_all[all_mask, :] = y[all_mask, :]

        # raw_y_train = rawlabels[train_mask, :]
        # raw_y_val = rawlabels[val_mask, :]
        # raw_y_test = rawlabels[test_mask, :]


        # return selected_idx, selected_idx_2
        # y_train:(235, 10), y_val:(235, 10), y_test:(235, 10)
        print('y_train:{}, y_val:{}, y_test:{}, y_all: {}'.format(y_train.shape,y_val.shape,y_test.shape, y_all.shape))
        print('train_mask:{}, val_mask:{}, test_mask:{}, all_mask: {}'.format(train_mask.shape,val_mask.shape,test_mask.shape, all_mask.shape))
        truefeatures_list = [truefeatures, truefeatures, truefeatures]
        return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all, all_mask



    def prepare_and_train(self, name = 'zhigang_zeng'):

        self.name = name
        # loadData(name)
        rawFeatures, labels, pids, rawlabels = self.loadFeature(name)
        print ("rawlabes: ", rawlabels)

        PAP, PSP, pid2idx, idx2pid = self.constructAdj(pids)

        PAP = self.loadPAP(PAP, pid2idx, name)
        PSP = self.loadPSP(PSP, pid2idx, name)

        N = len(pids)
        X_train, X_val, X_test, Allidx = self.constructIdx(list(range(N)), labels)

        #  truelabels, truefeatures, PAP, PSP, train_idx, val_idx, test_idx, allIdx

        adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all, all_mask = self.load_data_dblp(labels, rawlabels,  rawFeatures, PAP, PSP, X_train, X_val, X_test, Allidx)
        print (test_mask)
        print (all_mask)
        print (y_all)
        prec, rec, f1 = self.train(adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all, all_mask, rawlabels, needtSNE=True, rawFeature=rawFeatures)
        # print ("labels: ", rawlabels)
        print ("set of labels: ", len(set(rawlabels)))
        return prec, rec, f1

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)






    def getLabel(self, Y):
        Tlabels = self.enc.inverse_transform(Y)
        labels = [T[1] for T in Tlabels]
        return labels, len(set(labels))

    def checkGraph(self, adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all, all_mask, needtSNE=False, rawFeature=[]):

        prec, rec, f1 = 0.0, 0.0, 0.0
        nb_nodes = fea_list[0].shape[0]
        ft_size = fea_list[0].shape[1]
        nb_classes = y_train.shape[1]

        # adj = adj.todense()

        # features = features[np.newaxis]  # [1, nb_node, ft_size]
        fea_list = [fea[np.newaxis] for fea in fea_list]
        adj_list = [adj[np.newaxis] for adj in adj_list]


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
                                                               mp_att_size=200,
                                                               residual=residual, activation=nonlinearity)
        return logits, final_embedding, att_val


    def train(self, adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all, all_mask, rawlabels, needtSNE=False, rawFeature=[]):

        prec, rec, f1 = 0.0, 0.0, 0.0
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
        y_all = y_all[np.newaxis]

        train_mask = train_mask[np.newaxis]
        val_mask = val_mask[np.newaxis]
        test_mask = test_mask[np.newaxis]
        all_mask = all_mask[np.newaxis]

        biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

        print('build graph...')
        with tf.Graph().as_default():
            with tf.name_scope('input'):
                metric_ftr_in = tf.placeholder(dtype=tf.float32, shape=(nb_nodes, ft_size), name='metric_ftr_in')
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
            logits, final_embedding, att_val, centers_embed = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                               attn_drop, ffd_drop,
                                                               bias_mat_list=bias_in_list,
                                                               hid_units=hid_units, n_heads=n_heads, features=fea_list, labels=rawlabels,
                                                               residual=residual, activation=nonlinearity, feature_size=ft_size)


            # final_embedding: checkout Tensor("Sum:0", shape=(286, 64), dtype=float32)

            # logits: checkout Tensor("ExpandDims_3:0", shape=(1, 286, 30), dtype=float32)

            # cal masked_loss
            # lab_list = tf.placeholder(dtype=tf.float32, shape=(nb_nodes, ), name='lab_list')
            # ftr_resh = tf.placeholder(dtype=tf.float32, shape=(nb_nodes, ft_size), name='ftr_resh')
            log_resh = tf.reshape(logits, [-1, nb_classes])
            lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
            msk_resh = tf.reshape(msk_in, [-1])


            print ("final_embedding: checkout", final_embedding)
            print ("logits: checkout", logits)
            print ("log_resh: checkout", log_resh)
            # print ("ftr_resh: ", ftr_resh)
            print ("lab_resh: ", lab_resh)
            print ("fea_list: ", fea_list)
            print ("centers_embed: ", centers_embed)
            print ("batch_size, nb_nodes, nb_classes, ft_size", batch_size, nb_nodes, nb_classes, ft_size)

            loss = OSM_CAA_Loss(batch_size=nb_nodes)
            osm_loss = loss.forward

            # final_embedding: checkout Tensor("Sum:0", shape=(286, 64), dtype=float32)
            # logits: checkout Tensor("ExpandDims_3:0", shape=(1, 286, 30), dtype=float32)
            # log_resh: checkout Tensor("Reshape:0", shape=(286, 30), dtype=float32)
            # ftr_resh:  Tensor("ftr_resh:0", shape=(286, 100), dtype=float32)
            # lab_resh:  Tensor("Reshape_1:0", shape=(286, 30), dtype=int32)

            osmLoss, checkvalue = osm_loss(metric_ftr_in, rawlabels, centers_embed)
            SoftMaxloss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
            # loss = SoftMaxloss + osmLoss
            # 为什么loss会固定
            loss = osmLoss
            # loss = SoftMaxloss

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
                               metric_ftr_in: rawFeature,
                               is_train: True,
                               attn_drop: 0.6,
                               ffd_drop: 0.6}
                        fd = fd1
                        fd.update(fd2)
                        fd.update(fd3)
                        _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                           feed_dict=fd)
                        test_check_value = sess.run(checkvalue, feed_dict=fd)
                        print ("test_check_value: ", test_check_value)

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
                               metric_ftr_in: rawFeature,
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
                    print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f | vl_step: %d, tr_step: %d' %
                          (train_loss_avg / tr_step, train_acc_avg / tr_step,
                           val_loss_avg / vl_step, val_acc_avg / vl_step, vl_step, tr_step))

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
                    fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                           for i, d in zip(ftr_in_list, fea_list)}
                    fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                           for i, d in zip(bias_in_list, biases_list)}
                    fd3 = {lbl_in: y_all[ts_step * batch_size:(ts_step + 1) * batch_size],
                           msk_in: all_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                           metric_ftr_in: rawFeature,
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

                xx = np.expand_dims(jhy_final_embedding, axis=0)[all_mask]
                yy = y_all[all_mask]


                print ("check fd")
                print (fd)
                print ("XX: ", xx)
                print ("YY: ", yy)
                print('xx: {}, yy: {}, ts_size: {}, ts_step: {}, batch_size: {}'.format(xx.shape, yy.shape, ts_size, ts_step,batch_size))

                labels, numberofLabels = self.getLabel(yy)

                from utils import  clustering, pairwise_precision_recall_f1

                clusters_pred = clustering(xx, num_clusters=numberofLabels)
                prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
                print ('prec: ', prec, ', rec: ', rec, ', f1: ', f1, ', originNumberOfClusterlabels: ', numberofLabels)

                if needtSNE:
                    tSNEAnanlyse(xx, labels, join(settings.PIC_DIR, "HAN", "rawReature_%s_final.png" % (self.name)))
                    tSNEAnanlyse(rawFeature, labels, join(settings.PIC_DIR, "HAN", "rawReature_%s_features.png" % (self.name)))
                    tSNEAnanlyse(xx, clusters_pred, join(settings.PIC_DIR, "HAN", "rawReature_%s_result_label.png" % (self.name)))

                # my_KNN(xx, yy)
                # my_Kmeans(xx, yy)

                sess.close()

        return prec, rec, f1

if __name__ == '__main__':
    pass
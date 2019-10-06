# coding=utf-8
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from model.layers import GraphConvolution, InnerProductDecoder
import itertools
from tensorflow.contrib.layers import l2_regularizer
import keras

flags = tf.app.flags
FLAGS = flags.FLAGS



class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class DualGCNGraphFusion(Model):
    def __init__(self,  placeholders, num_features, num_nodes, num_logits, features_nonzero=None, **kwargs):
        super(DualGCNGraphFusion, self).__init__(**kwargs)

        self.num_logits = num_logits
        self.epoch = 0
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.graph1 = placeholders['graph1']
        self.graph2 = placeholders['graph2']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.epoch = placeholders['epoch']
        self.clusterEpoch = placeholders['clusterEpoch']
        # self.Featureinput = placeholders['Featureinput']
        self.build()

    def _build(self):

        # First GCN auto-encoder
        self.hidden_1_1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.graph1,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        # First GCN auto-encoder
        self.hidden_1_2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=FLAGS.hidden2,
                                        adj=self.graph1,
                                        act=lambda x: x,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.hidden_1_1)

        self.log_std_1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.graph1,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden_1_1)


        # # Second GCN auto-encoder
        self.hidden_2_1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.graph2,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        # # Second GCN auto-encoder
        self.hidden_2_2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=FLAGS.hidden2,
                                        adj=self.graph2,
                                        act=lambda x: x,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.hidden_2_1)

        self.log_std_2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.graph2,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden_2_1)

        # Fusion, 非线性的融合，(286 * 64)
        # self.w1 = tf.Variable(tf.random_normal([self.,10]))
        # self.b1 = tf.Variable(tf.zeros([1,10]) + 0.1)
        # self.z_3_mean = 0.5 * tf.add(self.hidden_1_2, self.hidden_2_2)
        # self.z_3_log_std =  0.5 * tf.add(self.log_std_1, self.log_std_2)
        self.hidden_1_2 = tf.cast(self.hidden_1_2, dtype=tf.float32)
        self.hidden_2_2 = tf.cast(self.hidden_2_2, dtype=tf.float32)
        # self.z_3_mean = tf.concat([self.hidden_1_2, self.hidden_2_2], axis=1)
        self.z_3_mean = tf.add(self.hidden_1_1, self.hidden_1_2)

        self.log_std_1 = tf.cast(self.log_std_1, dtype=tf.float32)
        self.log_std_2 = tf.cast(self.log_std_2, dtype=tf.float32)
        # self.z_3_log_std = tf.concat([self.log_std_1, self.log_std_2], axis=1)
        self.z_3_log_std = tf.add(self.log_std_1, self.log_std_2)

        # self.z_3_log_std = tf.layers.dense(self.z_3_mean , FLAGS.hidden2)

        self.z = self.z_3_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_3_log_std)  # element-wise


        self.reconstructions_1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z)

        self.reconstructions_2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z)


    def get_center_loss(self, features, labels, alpha, num_classes):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

        Return：
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        with tf.variable_scope(self.name):
            centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        return loss, centers, centers_update_op

    def pairwise_distance(self, fea, com):
        dims = len(fea.get_shape())
        if dims == 2:
            fea_k = fea[com[0], :]
            fea_j = fea[com[1], :]
        else:
            print("Please check the feature dimensions")
            return
        fea_k = tf.expand_dims(fea_k, 0)
        fea_j = tf.expand_dims(fea_j, 0)
        k_l2norm = tf.nn.l2_normalize(fea_k, 1)  ###pay attention to aixs
        j_l2norm = tf.nn.l2_normalize(fea_j, 1)
        loss_term = tf.reduce_sum(tf.multiply(k_l2norm, j_l2norm)) + 1
        # grad_term = k_l2norm/tf.norm(fea_j,ord=2) - k_l2norm*tf.square(j_l2norm)/tf.norm(fea_j,ord=2)
        return loss_term

    def pairwise_grad(self, fea, com):
        dims = len(fea.get_shape())
        if dims == 2:
            fea_k = fea[com[0], :]
            fea_j = fea[com[1], :]
        else:
            print("Please check the feature dimensions")
            return
        fea_k = tf.expand_dims(fea_k, 0)
        fea_j = tf.expand_dims(fea_j, 0)
        k_l2norm = tf.nn.l2_normalize(fea_k)
        j_l2norm = tf.nn.l2_normalize(fea_j)
        grad_term = k_l2norm / tf.norm(fea_j, ord=2) - k_l2norm * tf.square(j_l2norm) / tf.norm(fea_j, ord=2)
        return grad_term

    def get_island_loss(self, features, labels, alpha,alpha1, num_classes):
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        with tf.variable_scope(self.name):
            centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(minval=0, maxval=1, seed=None, dtype=tf.float32), trainable=False)

        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss_part1 = tf.nn.l2_loss(features - centers_batch)

        ####add new code
        index = np.arange(num_classes)
        print ('index: ', index, 'number_class: ', num_classes)
        combination = itertools.permutations(index, 2)

        pairwise_grad_val = {}
        pair_distance_loss = []
        for idx, item in enumerate(combination):
            index = idx / (num_classes - 1)
            lc_grad = self.pairwise_grad(centers_batch, item)
            if idx % (num_classes - 1) == 0:
                if index in pairwise_grad_val:
                    pairwise_grad_val[index] += lc_grad
                else:
                    pairwise_grad_val[index] = lc_grad
            else:
                if index in pairwise_grad_val:
                    pairwise_grad_val[index] += lc_grad
                else:
                    pairwise_grad_val[index] = lc_grad
            pair_distance_loss.append(self.pairwise_distance(centers_batch, item))

        print ('pair_distance_loss end')

        grad_pairwise = []
        for idx in range(num_classes):
            grad_pairwise.append(pairwise_grad_val[idx])

        grad_pairwise = tf.convert_to_tensor(grad_pairwise)
        grad_pairwise_batch = tf.gather(grad_pairwise, labels)
        grad_pairwise_batch = tf.squeeze(grad_pairwise_batch, axis=1)
        pair_distance_loss = tf.reduce_sum(pair_distance_loss)
        loss = alpha*loss_part1 + alpha1 * pair_distance_loss


        ####new code end
        print ('get_island_loss ending .... ')


        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)

        diff = diff + alpha1 * grad_pairwise_batch / (num_classes - 1)

        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)



        return loss, centers, centers_update_op, loss_part1, pair_distance_loss



# coding=utf-8
import tensorflow as tf
import numpy as np
from model.layers import GraphConvolution, InnerProductDecoder
import itertools
from tensorflow.contrib.layers import l2_regularizer


flags = tf.app.flags
FLAGS = flags.FLAGS




class OptimizerDualGCNAutoEncoder(object):

    def initVariable(self):
        FLAGS.graph1Variable = 1.0
        FLAGS.graph2Variable = 1.0
        FLAGS.KLlossVariable = 0.01
        FLAGS.CenterLossVariable = 10

    def getVariable(self, operationName, epoch=0, clusterEpoch=1):
        propotion = (1.0 * epoch / float(FLAGS.epochs)) * (1.0 * epoch / float(FLAGS.epochs)) * (
                    1.0 * epoch / float(FLAGS.epochs))
        if operationName == "graph1Variable":
            # return FLAGS.graph1Variable
            return FLAGS.graph1Variable - propotion * FLAGS.graph1Variable
        elif operationName == "graph2Variable":
            # return FLAGS.graph2Variable
            return FLAGS.graph2Variable - propotion * FLAGS.graph2Variable
        elif operationName == "KLlossVariable":
            # return FLAGS.KLlossVariable
            return FLAGS.KLlossVariable - propotion * FLAGS.KLlossVariable
        elif operationName == "ReconstructVariable":
            # return FLAGS.CenterLossVariable
            return FLAGS.ReconstructVariable - propotion * FLAGS.ReconstructVariable
        elif operationName == "CenterLossVariable":
            # return FLAGS.CenterLossVariable
            return 1.0 * FLAGS.CenterLossVariable * propotion * (1.0 * clusterEpoch / FLAGS.clusterEpochs)
        elif operationName == "SoftmaxVariable":
            # return FLAGS.CenterLossVariable
            return 1.0 * FLAGS.SoftmaxVariable * propotion
        else:
            return -1

    def  __init__(self, model, num_nodes, z_label, name, graph1, graph2):
        self.name = name

        self.epoch = 0
        self.num_nodes = num_nodes

        # 计算 Loss = loss1 + loss2 + KL-loss + island loss
        self.centerloss, self.centers, self.centers_update_op = self.CenterLoss(model, z_label)
        self.centerloss = self.centerloss * self.getVariable('CenterLossVariable', model.epoch)


        # 计算 reconstructLoss

        self.kl = self.kl_loss2(model.z_3_log_std, model.z_3_mean)
        self.reconstructloss1 =  self.getReconstructLoss(model.reconstructions_1, graph1['norm'], graph1['pos_weight'], graph1['labels'])
        self.reconstructloss2 =  self.getReconstructLoss(model.reconstructions_2, graph2['norm'], graph2['pos_weight'], graph2['labels'])
        self.reconstructloss = self.getVariable('ReconstructVariable', model.epoch) * (self.reconstructloss1 + self.reconstructloss2 + 2.0 * self.kl)


        self.cost = self.reconstructloss + self.centerloss

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.DGAE_learning_rate)

        with tf.control_dependencies([self.centers_update_op]):
            self.opt_op = self.optimizer.minimize(self.cost)

        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # 这个accuracy没啥用
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(model.y, 1), z_label), tf.float32))

    def CenterLoss(self, model, z_label, alpha=1.5, alpha1=1.0):
        # loss, centers, centers_update_op, loss_part1, pair_distance_loss = model.get_island_loss(model.z_3, z_label, alpha, alpha1, len(set(z_label)))
        loss, centers, centers_update_op = model.get_center_loss(model.z, z_label, alpha, len(set(z_label)))
        return loss, centers, centers_update_op

    def getReconstructLoss(self, preds_sub , norm, pos_weight, labels):
        # reconstrcut loss
        return norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels, pos_weight=pos_weight))


    def Cost(self, labels, model):

        # self.softmax_loss = FLAGS.SoftmaxVariable * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model.y))
        # self.softmax_loss = self.getVariable('SoftmaxVariable', model.epoch) * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model.y))

        # self.l2_loss
        # self.L2loss = l2_regularizer(scale=FLAGS.L2Scale)(model.z_3)

        # KL loss
        self.loss3 = self.getVariable('KLlossVariable', model.epoch) * (self.kl_divergence(model.z_3_log_std, model.z_mean_1) + self.kl_divergence(model.z_3, model.z_mean_2) )


        # return self.loss1 +  self.loss2
        # return  self.softmax_loss + self.loss3
        return self.loss3

    def SpecialLog(self, y):
        return tf.log(tf.clip_by_value(y,1e-8,1.0))

    def kl_loss2(self, log_std, mean):
        return -1.0 *(0.5 / self.num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * log_std - tf.square(mean) - tf.square(tf.exp(log_std)), 1))

    def kl_divergence(self, p, q):
        # return tf.log(p)
        # return self.KLloss(p,q)
        return tf.abs(tf.reduce_sum(p * (self.SpecialLog(p) - self.SpecialLog(q))))

    def KLloss(self, layerA, layerB):
        kl = (0.5 / self.num_nodes) * tf.reduce_mean(
            tf.reduce_sum(1 + 2 * layerA - tf.square(layerB) - tf.square(tf.exp(layerA)), 1))
        return kl

    def VAEloss(self, preds, labels, pos_weight, norm, z_mean, z_log_std):
        cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=pos_weight))
        kl = self.KLloss(z_log_std, z_mean)
        # kl = (0.5 / self.num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * z_log_std - tf.square(z_mean) - tf.square(tf.exp(z_log_std)), 1))
        cost -= kl
        return cost



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
                                        act=tf.nn.relu,
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
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.hidden_2_1)

        # Fusion, 非线性的融合，(286 * 64)
        self.z_3_add = tf.add(self.hidden_1_2, self.hidden_2_2)
        self.z_3 = tf.layers.dense(self.z_3_add , FLAGS.hidden2, activation=tf.tanh)


        # Variable layout
        self.z_3_mean = tf.layers.dense(self.z_3,  FLAGS.hidden2)
        self.z_3_log_std = tf.layers.dense(self.z_3,  FLAGS.hidden2)

        self.z = self.z_3_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_3_log_std)  # element-wise


        self.reconstructions_1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z)

        self.reconstructions_2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z)



        # Y
        # self.y = tf.layers.dense(self.z_3, self.num_logits)

        # print ('debuging ', self.y)

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


# coding=utf-8
import tensorflow as tf
from model.layers import GraphConvolution, InnerProductDecoder

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerDualGCNAutoEncoder(object):

    def __init__(self, preds_1, labels_1, preds_2, labels_2, model, num_nodes, pos_weight, norm):
        self.num_nodes = num_nodes

        # 计算 Loss = loss1 + loss2 + KL-loss + island loss
        self.cost = self.Cost(preds_1, labels_1, preds_2, labels_2, model, pos_weight, norm)
        print ('cost: ', self.cost)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # 这个accuracy没啥用
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_1), 0.5), tf.int32),
                                           tf.cast(labels_1, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def Cost(self, preds_1, labels_1, preds_2, labels_2, model, pos_weight, norm):
        # loss1
        self.loss1 = FLAGS.graph1Variable * (self.VAEloss(preds_1, labels_1, pos_weight, norm, model.z_mean_1, model.z_log_std_1))

        # loss2
        self.loss2 = FLAGS.graph2Variable * (self.VAEloss(preds_2, labels_2, pos_weight, norm, model.z_mean_2, model.z_log_std_2))

        # KL loss
        self.loss3 = FLAGS.KLlossVariable * (self.kl_divergence(model.z_3, model.z_mean_1) + self.kl_divergence(model.z_3, model.z_mean_2))
        return self.loss1 +  self.loss2 + self.loss3
        # return loss3

    def SpecialLog(self, y):
        return tf.log(tf.clip_by_value(y,1e-8,1.0))

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

class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.preds = preds
        self.labels = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)


        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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
    def __init__(self,  placeholders, num_features, num_nodes, features_nonzero=None, **kwargs):
        super(DualGCNGraphFusion, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.graph1 = placeholders['graph1']
        self.graph2 = placeholders['graph2']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):

        # First GCN auto-encoder
        self.hidden_1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.graph1,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.z_mean_1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.graph1,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden_1)

        self.z_log_std_1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.graph1,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden_1)

        self.z_1 = self.z_mean_1 + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std_1)  # element-wise

        self.reconstructions_1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z_1)

        # # Second GCN auto-encoder
        self.hidden_2 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.graph2,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.z_mean_2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.graph2,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden_2)

        self.z_log_std_2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.graph2,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden_2)

        self.z_2 = self.z_mean_2 + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std_2)  # element-wise

        self.reconstructions_2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z_2)

        # Fusion, 线性的融合，(286 * 64)
        self.z_3_temp = tf.add(self.z_mean_1, self.z_mean_2)
        self.z_3 = tf.layers.dense(self.z_3_temp , FLAGS.hidden2)







class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero=None, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        '''
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        '''

        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)  # element-wise

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z)

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


class OptimizerDualGCNAutoEncoder(object):

    def initVariable(self):
        FLAGS.graph1Variable = 1.0
        FLAGS.graph2Variable = 1.0
        FLAGS.KLlossVariable = 0.01
        FLAGS.CenterLossVariable = 10

    def getVariable(self, operationName, epoch=0, clusterEpoch=1):
        if operationName == "graph1Variable":
            # return FLAGS.graph1Variable
            return FLAGS.graph1Variable
        elif operationName == "graph2Variable":
            # return FLAGS.graph2Variable
            return FLAGS.graph2Variable
        elif operationName == "KLlossVariable":
            # return FLAGS.KLlossVariable
            return FLAGS.KLlossVariable
            # return FLAGS.KLlossVariable - propotion * FLAGS.KLlossVariable
        elif operationName == "ReconstructVariable":
            # return FLAGS.CenterLossVariable
            return FLAGS.ReconstructVariable
        elif operationName == "CenterLossVariable":
            return FLAGS.CenterLossVariable
            # return 1.0 * FLAGS.CenterLossVariable * (1.0 * clusterEpoch / FLAGS.clusterEpochs)
            # return  FLAGS.CenterLossVariable
        elif operationName == "SoftmaxVariable":
            # return FLAGS.CenterLossVariable
            return  FLAGS.SoftmaxVariable
        else:
            return -1

    # def getVariable(self, operationName, epoch=0, clusterEpoch=1):
    #     propotion = (1.0 * epoch / float(FLAGS.epochs)) * (1.0 * epoch / float(FLAGS.epochs)) * (
    #                 1.0 * epoch / float(FLAGS.epochs))
    #     if operationName == "graph1Variable":
    #         # return FLAGS.graph1Variable
    #         return FLAGS.graph1Variable - propotion * FLAGS.graph1Variable
    #     elif operationName == "graph2Variable":
    #         # return FLAGS.graph2Variable
    #         return FLAGS.graph2Variable - propotion * FLAGS.graph2Variable
    #     elif operationName == "KLlossVariable":
    #         # return FLAGS.KLlossVariable
    #         return 1.0 * FLAGS.KLlossVariable * propotion
    #         # return FLAGS.KLlossVariable - propotion * FLAGS.KLlossVariable
    #     elif operationName == "ReconstructVariable":
    #         # return FLAGS.CenterLossVariable
    #         return FLAGS.ReconstructVariable - propotion * FLAGS.ReconstructVariable
    #     elif operationName == "CenterLossVariable":
    #         # return FLAGS.CenterLossVariable
    #         return 1.0 * FLAGS.CenterLossVariable * propotion * (1.0 * clusterEpoch / FLAGS.clusterEpochs)
    #     elif operationName == "SoftmaxVariable":
    #         # return FLAGS.CenterLossVariable
    #         return 1.0 * FLAGS.SoftmaxVariable * propotion
    #     else:
    #         return -1

    def  __init__(self, model, num_nodes, z_label, name, graph1, graph2):
        self.name = name

        self.epoch = 0
        self.num_nodes = num_nodes

        # 计算 Loss = loss1 + loss2 + KL-loss + island loss
        self.centerloss, self.centers, self.centers_update_op = self.CenterLoss(model, z_label, alpha=FLAGS.CenterLossVariable)
        self.centerloss = self.centerloss * self.getVariable('CenterLossVariable', model.epoch, model.clusterEpoch)


        # 计算 reconstructLoss


        self.kl = self.kl_loss2(model.z_3_log_std, model.z_3_mean)
        self.kl1 = self.kl_loss2(model.log_std_1, model.hidden_1_2)
        self.kl2 = self.kl_loss2(model.log_std_2, model.hidden_2_2)

        self.reconstructloss1 =  self.getReconstructLoss(model.reconstructions_1, graph1['norm'], graph1['pos_weight'], graph1['labels'])
        self.reconstructloss2 =  self.getReconstructLoss(model.reconstructions_2, graph2['norm'], graph2['pos_weight'], graph2['labels'])
        self.reconstructloss3 =  self.getReconstructLoss(model.reconstructions_3, graph1['norm'], graph1['pos_weight'], graph1['labels'])
        self.reconstructloss4 =  self.getReconstructLoss(model.reconstructions_4, graph2['norm'], graph2['pos_weight'], graph2['labels'])
        self.reconstructloss = self.getVariable('ReconstructVariable', model.epoch) * (self.reconstructloss1 + self.reconstructloss2 + 2.0 * self.kl )
        self.reconstructloss2 = self.getVariable('ReconstructVariable', model.epoch) * (self.reconstructloss3 + self.reconstructloss4 +  self.kl1 + self.kl2 )



        # 计算distribute loss
        # self.distributeLoss = self.getVariable('KLlossVariable', model.epoch) * (self.kl_divergence(model.z_3_mean, model.hidden_1_2) + self.kl_divergence(model.z_3_mean, model.hidden_2_2) )
        # self.distributeLoss = self.getVariable('KLlossVariable', model.epoch) * (self.kl_divergence(model.z_3_mean, model.hidden_1_2) + self.kl_divergence(model.z_3_mean, model.hidden_2_2) )

        # self.targetdistributionloss = FLAGS.finetuningVariable * (self.targetDistributionLoss(model.z_3_mean, self.centers))

        # Target Distribution Loss
        # self.targetdistributionloss = self.targetDistributionLoss(model.z_3_mean, self.centers)

        # self.l2_loss
        # self.L2loss = l2_regularizer(scale=FLAGS.L2Scale)(model.z)

        # self.cost = self.reconstructloss + self.centerloss + self.distributeLoss + self.targetdistributionloss
        # self.cost = self.reconstructloss + self.distributeLoss + self.targetdistributionloss
        self.cost = self.reconstructloss + self.centerloss

        # self.optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.Finetuning_learning_rate)
        # self.opt_op2 = self.optimizer2.minimize(self.targetdistributionloss)
        # self.grads_vars2 = self.optimizer2.compute_gradients(self.targetdistributionloss)


        self.optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.DGAE_learning_rate)

        with tf.control_dependencies([self.centers_update_op]):
            self.opt_op = self.optimizer.minimize(self.cost)

        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # 这个accuracy没啥用
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(model.y, 1), z_label), tf.float32))

    def CenterLoss(self, model, z_label, alpha=0.05, alpha1=1.0):
        # loss, centers, centers_update_op, loss_part1, pair_distance_loss = model.get_island_loss(model.z_3, z_label, alpha, alpha1, len(set(z_label)))
        loss, centers, centers_update_op = model.get_center_loss(model.z_3_mean, z_label, alpha, len(set(z_label)))
        # loss, centers, centers_update_op = model.get_center_loss(model.z_3_mean, z_label, alpha, len(set(z_label)))
        return loss, centers, centers_update_op

    def getReconstructLoss(self, preds_sub , norm, pos_weight, labels):
        # reconstrcut loss
        return norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels, pos_weight=pos_weight))

    def Cost(self, labels, model):

        # self.softmax_loss = FLAGS.SoftmaxVariable * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model.y))
        # self.softmax_loss = self.getVariable('SoftmaxVariable', model.epoch) * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model.y))

        # KL loss
        self.loss3 = self.getVariable('KLlossVariable', model.epoch) * (self.kl_divergence(model.z, model.hidden_1_2) + self.kl_divergence(model.z, model.hidden_2_2) )
        # self.loss3 = self.getVariable('KLlossVariable', model.epoch) * (self.kl_divergence(model.z_3_mean, model.hidden_1_2) + self.kl_divergence(model.z_3_mean, model.hidden_2_2) )


        # return self.loss1 +  self.loss2
        # return  self.softmax_loss + self.loss3
        return self.loss3


    def kl_loss2(self, log_std, mean):
        return -1.0 *(0.5 / self.num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * log_std - tf.square(mean) - tf.square(tf.exp(log_std)), 1))

    def SpecialLog(self, y):
        return tf.log(tf.clip_by_value(y,1e-8,1.0))

    def kl_divergence(self, p, q):
        return tf.reduce_sum(p * (self.SpecialLog(p) - self.SpecialLog(q)))
        # crossE = tf.nn.softmax_cross_entropy_with_logits(logits=pred_subj, labels=newY)
        # return accr_subj_test
        # y = p / q
        # return tf.reduce_sum(tf.multiply(p, self.SpecialLog(p) - self.SpecialLog(q)))
        # return tf.log(p)
        # return self.KLloss(p,q)

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


    def targetDistributionLoss(self,  features, centers, freedomAlpha=2.0):
        distributionA = self.tDistribution(features, centers, freedomAlpha)
        distributionB = self.auxiliaryDistriution(distributionA)

        print (distributionA)
        print (distributionB)
        return self.kl_divergence(distributionA, distributionB)
        # return tf.keras.losses.KLDivergence(distributionA, distributionB)
        # return tf.distributions.kl_divergence(distributionA, distributionB)

    # 计算t分布
    def tDistribution(self, features, centers, freedomAlpha=2.0):
        expanded_a = tf.expand_dims(features, 1)
        expanded_b = tf.expand_dims(centers, 0)
        distances = 1 + tf.square(tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)) / freedomAlpha
        distances **= -1.0 * (freedomAlpha + 1.0) / 2.0
        distances = tf.transpose(tf.transpose(distances) / tf.reduce_sum(distances, axis=1))
        return distances

    def auxiliaryDistriution(self, q):
        q **= 2.0
        f = tf.reshape(tf.reduce_sum(q, axis=1), (-1, 1))
        q2 = q / f
        Sum = tf.reduce_sum(q2, axis=0)
        p = q2 / Sum
        return p

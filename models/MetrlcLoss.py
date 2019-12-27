import tensorflow as tf
import random
import  numpy as np


# Implementation of Deep Metric Learning by Online Soft Mining and Class-Aware Attention
# https://arxiv.org/pdf/1811.01459v2.pdf

class OSM_CAA_Loss():
    def __init__(self, alpha=1.2, l=0.5, use_gpu=True, batch_size=32, beta=0.5):
        self.use_gpu = use_gpu
        self.alpha = 2.0  # margin of weighted contrastive loss, as mentioned in the paper
        self.l = 0.5  # hyperparameter controlling weights of positive set and the negative set
        self.osm_sigma = 0.8  # \sigma OSM (0.8) as mentioned in paper
        self.n = batch_size

    def safe_divisor(self, x):
        return  tf.clip_by_value(x, clip_value_min=tf.constant(1e-12),
                                clip_value_max=tf.constant(1e12))

    def pairwise_dist(self, A, B):
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          A,    [m,d] matrix
          B,    [n,d] matrix
        Returns:
          D,    [m,n] matrix of pairwise distances
        """

        print ("A: ", A)
        print ("B: ", B)

        with tf.variable_scope('pairwise_dist'):
            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(A), 1)
            nb = tf.reduce_sum(tf.square(B), 1)

            # na as a row and nb as a co"lumn vectors
            na = tf.reshape(na, [-1, 1])
            nb = tf.reshape(nb, [1, -1])

            # return pairwise euclidead difference matrix
            C = tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0)
            C = self.safe_divisor(C)
            D = tf.sqrt(C)
        return D

    def EuclideanDistance(self, A, B):
        C = tf.reduce_sum(tf.square(A - B), 1)
        C = self.safe_divisor(C)
        C = tf.reshape(C, [-1,1])
        return tf.sqrt(C)

    def forward(self, x, labels, embd):
        '''
        x : feature vector : (n x d)
        labels : (n,)
        embd : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        x = tf.math.l2_normalize(x, 1)
        n = self.n
        r = tf.ones([n, 1], tf.float32)

        print ("r: ", r)

        dist = self.safe_divisor(r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r))
        dist = self.safe_divisor(tf.math.sqrt(dist))
        dist = tf.clip_by_value(dist, clip_value_min=tf.constant(1e-12),
                                clip_value_max=tf.constant(1e12))  # 0 value sometimes becomes nan

        p_mask = tf.cast(tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :]), tf.float32)
        n_mask = 1 - p_mask

        S_ = tf.clip_by_value(tf.nn.relu(self.alpha - dist), clip_value_min=tf.constant(1e-12),clip_value_max=tf.constant(1e12))

        # history reason
        embd = tf.math.l2_normalize(embd, 0)

        CenterDistance = self.pairwise_dist(x, tf.transpose(embd)) # x: (n,d), embed(c,d), CenterDistance(n,m)
        denom = tf.reduce_sum(tf.exp(CenterDistance), 1)
        # num = tf.exp(tf.reduce_sum(x * tf.transpose(tf.gather(embd, labels, axis=1)), 1))
        PointDistance = self.EuclideanDistance(x , tf.transpose(tf.gather(embd, labels, axis=1)))
        # PointDistance = x * tf.transpose(tf.gather(embd, labels, axis=1))
        print ("PointDistance: ", PointDistance)
        num = tf.exp(tf.reduce_sum(PointDistance, 1))


        atten_class = num / denom
        temp = tf.tile(tf.expand_dims(atten_class, 0), [n, 1])
        A = tf.math.maximum(temp, tf.transpose(temp))


        W_P = A * p_mask
        W_N = A * n_mask
        W_P = W_P * (1 - tf.eye(n))
        W_N = W_N * (1 - tf.eye(n))

        # L_P = tf.reduce_mean(W_P * tf.pow(dist, 2)) / 2
        L_P = tf.reduce_sum(W_P * tf.pow(dist, 2)) / (2 * tf.reduce_sum(W_P))
        # L_N = tf.reduce_mean(W_N * tf.pow(S_, 2)) / 2
        L_N = tf.reduce_sum(W_N * tf.pow(S_, 2)) / (2 * tf.reduce_sum(W_N))
        # L_P = tf.reduce_sum(W_P * tf.pow(dist, 2)) / 2
        # L_N = tf.reduce_sum(W_N * tf.pow(S_, 2) ) / 2
        # L_P = tf.reduce_sum(W_P) / 2
        # L_N = tf.reduce_sum(W_N) / 2


        L = (1 - self.l) * L_P + self.l * L_N

        return L, [L_P, L_N]

if __name__ == '__main__':
    sess = tf.Session()
    x = tf.random.uniform([32, 200])  # (batch size= 32, embedding dim= 200)
    embd = tf.random.uniform([200, 10])  # (embedding dim= 200 , num of classes = 10)
    labels = np.random.choice(range(1, 10), size=32)

    loss = OSM_CAA_Loss()
    osm_loss = loss.forward

    loss_val = osm_loss(x, labels, embd)
    sess.run(loss_val)




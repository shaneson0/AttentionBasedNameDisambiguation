
# from os.path import join
# from utils import string_utils, settings
# import json
#
# authornames = ['Hongbin Li', 'Hua Bai', 'Kexin Xu', 'Lin Huang', 'Lu Han', 'Min Zheng', 'Qiang Shi', 'Rong Yu', 'Tao Deng', 'Wei Quan', 'Xu Xu', 'Yanqing Wang', 'Yong Tian']
#
# Len = len(authornames)
# for i in range(Len):
#     authornames[i] = string_utils.clean_name(authornames[i])
#
# with open(join(settings.DATA_DIR, 'test_name_list2.json'), 'w') as fp:
#     json.dump(authornames, fp)
#     fp.close()

# print(authornames)


import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
# from tfp.distributions import kl_divergence




# A = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
# B = tf.constant([1.0,2.0,3.0])
# C = A / tf.reshape(B, (-1, 1))
#
# sess = tf.Session()
# print (sess.run(C))

A = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
B = tf.constant([[1.0, 1.0], [2.0, 2.0]])


def tDistribution(A, B, freedomAlpha = 2.0):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(B, 0)
    distances = 1 + tf.square(tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)) / freedomAlpha
    distances **= -1.0 * (freedomAlpha + 1.0) / 2.0
    distances = K.transpose(K.transpose(distances) / K.sum(distances, axis=1))
    return distances

def auxiliaryDistriution(q):
    q **= 2.0
    f = tf.reshape(K.sum(q, axis=1), (-1,1))
    q2 = q / f
    Sum = K.sum(q2, axis=0)
    p = q2/Sum
    return p

sess = tf.Session()
C = tDistribution(A, B)
print (sess.run(C))


A = auxiliaryDistriution(C)

# print (C)
# print (A)
from sklearn.metrics import mutual_info_score
c = sess.run(C)
a = sess.run(A)

k = tf.keras.losses.KLDivergence()
loss = k(c,a)

print (sess.run(loss))
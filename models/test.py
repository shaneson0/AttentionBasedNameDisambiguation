import tensorflow as tf

def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
    with tf.variable_scope('pairwise_dist'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


if __name__ == '__main__':
    with tf.Session() as sess:
        A = tf.constant([[1,2], [2,3]], dtype=tf.float32)
        B = tf.constant([[4,5], [5,6], [7,8]], dtype=tf.float32)
        C = pairwise_dist(A,B)
        D = tf.reduce_sum(tf.exp(C), 1)
        resultC = sess.run([D])
        print ("C: ", resultC)
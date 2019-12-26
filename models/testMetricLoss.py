from HeterogeneousGraph import HAN
import tensorflow as tf
from models.MetrlcLoss import OSM_CAA_Loss


name = "hongbin_li"

han = HAN.HAN()
features, labels, pids, rawlabels = han.loadFeature(name)


print ("res: ", features)
print ("n_nodes: ", features.shape[0])


def buildModel():
    ftr_input = tf.placeholder("float", [None, 100])
    D1 = tf.layers.dense(ftr_input, 100, activation=tf.nn.sigmoid)
    D2 =  tf.layers.dense(D1, 100, activation=tf.nn.sigmoid)
    D3 =  tf.layers.dense(D2, 100, activation=tf.nn.sigmoid)
    return D3


def Optimized(model, nb_nodes):
    osm_caa_loss = OSM_CAA_Loss(batch_size=nb_nodes)
    osm_loss = osm_caa_loss.forward
    return osm_caa_loss

def training(loss, lr, l2_coef):
    # weight decay
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                       in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

    # optimizer
    opt = tf.train.AdamOptimizer(learning_rate=lr)

    # training op
    train_op = opt.minimize(loss + lossL2)

    return train_op




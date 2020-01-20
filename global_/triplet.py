from keras import backend as K


def l2Norm(x):
    return K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def global_triplet_loss(_, y_pred):

    margin = K.constant(1)
    L1 =  K.mean(K.maximum(K.constant(0),  K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))
    L2 =  K.mean(K.maximum(K.constant(0),  K.square(y_pred[:,2,0]) - K.square(y_pred[:,3,0]) + margin))
    L3 =  K.square(y_pred[:,4,0])
    # L3 =  K.mean(K.maximum(K.constant(0),  K.square(y_pred[:,4,0]) - K.square(y_pred[:,5,0]) + margin))

    return L1 + L2 + L3
    # return 0.6 * L1 + 0.2 * L2 + 0.2 * L3

def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])




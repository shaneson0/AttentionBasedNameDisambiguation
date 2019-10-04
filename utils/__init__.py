import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score

def lossPrint(x, loss1, loss2, loss3):

    plt.subplot(3, 1, 1)
    # plt.plot(x1, y1, 'o-',color='r')
    plt.plot(x, loss1, '.-', label="loss1")
    plt.legend(loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(x, loss2, '.-', label="loss2")
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.plot(x, loss3, '.-', label="loss3")
    plt.legend(loc='best')

    plt.show()

# setting for model
def getSetting():
    flags = tf.app.flags
    flags.DEFINE_float('DGAE_learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_float('Finetuning_learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 8000, 'Number of epochs to train.')
    flags.DEFINE_integer('finetuningEpochs', 8000, 'Number of epochs to train.')
    flags.DEFINE_integer('clusterEpochs', 3, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')  # 32
    flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')  # 16
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

    flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
    flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
    # flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
    flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

    flags.DEFINE_float('SoftmaxVariable', 0.3, 'Weight for softmax.')
    # flags.DEFINE_float('KLlossVariable', 0.005, 'Weight for KL loss on graph comparing.')
    # flags.DEFINE_float('KLlossVariable', 0.0001, 'Weight for KL loss on graph comparing.')
    flags.DEFINE_float('KLlossVariable', 1.0 , 'Weight for KL loss on graph comparing.')
    flags.DEFINE_float('finetuningVariable', 0.001, 'fine tune variable')
    # flags.DEFINE_float('CenterLossVariable', 0.4, 'Weight for the cluster loss --- CenterLoss .')
    flags.DEFINE_float('CenterLossVariable', 0.0, 'Weight for the cluster loss --- CenterLoss .')
    flags.DEFINE_float('ReconstructVariable', 2, 'Weight for the cluster loss --- CenterLoss .')
    flags.DEFINE_float('L2Scale', 0.001, 'Weight for L2 regular')

    return flags


from sklearn.manifold import TSNE
def tSNEAnanlyse(emb, labels, savepath=False):
    plt.figure()
    labels = np.array(labels) + 2
    print('labels : ', labels)
    print('labels type: ', len(set(labels)))
    X_new = TSNE(learning_rate=100).fit_transform(emb)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, marker='o')
    plt.show()

    if savepath:
        plt.savefig(savepath)
    plt.close()

def sNEComparingAnanlyse(emb, cureentLabels, TureLabels, oldLabels, savepath=False):
    plt.figure()
    X_new = TSNE(learning_rate=100).fit_transform(emb)
    plt.subplot(3,1,1)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=cureentLabels, marker='o')
    plt.subplot(3,1,2)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=TureLabels, marker='o')
    plt.subplot(3,1,3)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=oldLabels, marker='o')
    plt.show()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def PCAAnanlyse(emb, labels):
    labels = np.array(labels) + 2
    print('labels : ', labels)
    print('labels type: ', len(set(labels)))
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(emb)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, marker='o')
    plt.show()


def clustering(embeddings, num_clusters):
    model = AgglomerativeClustering(n_clusters=num_clusters).fit(embeddings)
    return model.labels_



def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)

def get_hidden_output(model, inp):
    get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.layers[5].get_output_at(0), ])
    activations = get_activations([inp, 0])
    return activations[0]


def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb-test_embs[0])
    score2 = np.linalg.norm(anchor_emb-test_embs[1])
    return [score1, score2]


def full_auc(model, test_triplets):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    grnds = []
    preds = []
    preds_before = []
    embs_anchor, embs_pos, embs_neg = test_triplets

    inter_embs_anchor = get_hidden_output(model, embs_anchor)
    inter_embs_pos = get_hidden_output(model, embs_pos)
    inter_embs_neg = get_hidden_output(model, embs_neg)
    # print(inter_embs_pos.shape)

    accs = []
    accs_before = []

    for i, e in enumerate(inter_embs_anchor):
        if i % 10000 == 0:
            print('test', i)

        emb_anchor = e
        emb_pos = inter_embs_pos[i]
        emb_neg = inter_embs_neg[i]
        test_embs = np.array([emb_pos, emb_neg])

        emb_anchor_before = embs_anchor[i]
        emb_pos_before = embs_pos[i]
        emb_neg_before = embs_neg[i]
        test_embs_before = np.array([emb_pos_before, emb_neg_before])

        predictions = predict(emb_anchor, test_embs)
        predictions_before = predict(emb_anchor_before, test_embs_before)

        acc_before = 1 if predictions_before[0] < predictions_before[1] else 0
        acc = 1 if predictions[0] < predictions[1] else 0
        accs_before.append(acc_before)
        accs.append(acc)

        grnd = [0, 1]
        grnds += grnd
        preds += predictions
        preds_before += predictions_before

    auc_before = roc_auc_score(grnds, preds_before)
    auc = roc_auc_score(grnds, preds)
    print('test accuracy before', np.mean(accs_before))
    print('test accuracy after', np.mean(accs))

    print('test AUC before', auc_before)
    print('test AUC after', auc)
    return auc

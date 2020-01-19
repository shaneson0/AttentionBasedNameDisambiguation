from os.path import join
import os
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from global_.triplet import l2Norm, euclidean_distance, triplet_loss, accuracy, global_triplet_loss
from global_.embedding import EMB_DIM
from utils import eval_utils
from utils import data_utils
from utils import settings

"""
global_ metric learning model
"""

#
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# anchor_input (InputLayer)       (None, 100)          0
# __________________________________________________________________________________________________
# pos_input (InputLayer)          (None, 100)          0
# __________________________________________________________________________________________________
# neg_input (InputLayer)          (None, 100)          0
# __________________________________________________________________________________________________
# first_emb_layer (Dense)         (None, 128)          12928       anchor_input[0][0]
#                                                                  pos_input[0][0]
#                                                                  neg_input[0][0]
# __________________________________________________________________________________________________
# last_emb_layer (Dense)          (None, 64)           8256        first_emb_layer[0][0]
#                                                                  first_emb_layer[1][0]
#                                                                  first_emb_layer[2][0]
# __________________________________________________________________________________________________
# norm_layer (Lambda)             (None, 64)           0           last_emb_layer[0][0]
#                                                                  last_emb_layer[1][0]
#                                                                  last_emb_layer[2][0]
# __________________________________________________________________________________________________
# pos_dist (Lambda)               (None, 1)            0           norm_layer[0][0]
#                                                                  norm_layer[1][0]
# __________________________________________________________________________________________________
# neg_dist (Lambda)               (None, 1)            0           norm_layer[0][0]
#                                                                  norm_layer[2][0]
# __________________________________________________________________________________________________
# stacked_dists (Lambda)          (None, 2)            0           pos_dist[0][0]
#                                                                  neg_dist[0][0]
# ==================================================================================================
# Total params: 21,184
# Trainable params: 21,184
# Non-trainable params: 0
# __________________________________________________________________________________________________

class GlobalTripletModel:

    def __init__(self, data_scale):
        self.data_scale = data_scale
        self.train_triplets_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.data_scale))
        self.test_triplets_dir = join(settings.OUT_DIR, 'test-triplets')
        self.train_triplet_files_num = self.get_triplets_files_num(self.train_triplets_dir)
        self.test_triplet_files_num = self.get_triplets_files_num(self.test_triplets_dir)
        print('test file num', self.test_triplet_files_num)

    @staticmethod
    def get_triplets_files_num(path_dir):
        files = []
        for f in os.listdir(path_dir):
            if f.startswith('anchor_embs_'):
                files.append(f)
        return len(files)

    def load_batch_triplets(self, f_idx, role='train'):
        if role == 'train':
            cur_dir = self.train_triplets_dir
        else:
            cur_dir = self.test_triplets_dir
        X1 = data_utils.load_data(cur_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
        X2 = data_utils.load_data(cur_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
        X3 = data_utils.load_data(cur_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        X4 = data_utils.load_data(cur_dir, 'atten_embs_{}_{}.pkl'.format(role, f_idx))
        X5 = data_utils.load_data(cur_dir, 'atten_pos_{}_{}.pkl'.format(role, f_idx))
        X6 = data_utils.load_data(cur_dir, 'atten_neg_{}_{}.pkl'.format(role, f_idx))

        return X1, X2, X3, X4, X5, X6

    def load_triplets_data(self, role='train'):
        X1 = np.empty([0, EMB_DIM])
        X2 = np.empty([0, EMB_DIM])
        X3 = np.empty([0, EMB_DIM])
        X4 = np.empty([0, EMB_DIM])
        X5 = np.empty([0, EMB_DIM])
        X6 = np.empty([0, EMB_DIM])

        if role == 'train':
            f_num = self.train_triplet_files_num
        else:
            f_num = self.test_triplet_files_num
        for i in range(f_num):
            print('load', i)
            x1_batch, x2_batch, x3_batch, x4_batch, x5_batch, x6_batch = self.load_batch_triplets(i, role)
            p = np.random.permutation(len(x1_batch))
            x1_batch = np.array(x1_batch)[p]
            x2_batch = np.array(x2_batch)[p]
            x3_batch = np.array(x3_batch)[p]
            x4_batch = np.array(x4_batch)[p]
            x5_batch = np.array(x5_batch)[p]
            x6_batch = np.array(x6_batch)[p]

            X1 = np.concatenate((X1, x1_batch))
            X2 = np.concatenate((X2, x2_batch))
            X3 = np.concatenate((X3, x3_batch))
            X4 = np.concatenate((X4, x4_batch))
            X5 = np.concatenate((X5, x5_batch))
            X6 = np.concatenate((X6, x6_batch))

        return X1, X2, X3, X4, X5, X6

    @staticmethod
    def create_triplet_model():
        emb_anchor = Input(shape=(EMB_DIM, ), name='anchor_input')
        emb_pos = Input(shape=(EMB_DIM, ), name='pos_input')
        emb_neg = Input(shape=(EMB_DIM, ), name='neg_input')
        emb_atten = Input(shape=(EMB_DIM, ), name='attention_input')
        emb_atten_pos = Input(shape=(EMB_DIM, ), name='attention_input_posive')
        emb_atten_neg = Input(shape=(EMB_DIM, ), name='attention_input_negive')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')

        layer1_Atten = Dense(128, activation='relu', name='first_emb_layer')
        layer2_Atten = Dense(64, activation='relu', name='last_emb_layer')

        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))


        encoded_emb_atten = norm_layer(layer2_Atten(layer1_Atten(emb_atten)))
        encoded_emb_atten_pos = norm_layer(layer2_Atten(layer1_Atten(emb_atten_pos)))
        encoded_emb_atten_neg = norm_layer(layer2_Atten(layer1_Atten(emb_atten_neg)))

        Trans = Dense(64, activation=None, name='Anchor')
        Anchor = Trans([encoded_emb, encoded_emb_atten])
        Positive = Trans([encoded_emb_pos, encoded_emb_atten_pos])
        Negative = Trans([encoded_emb_neg, encoded_emb_atten_neg])

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([Anchor, Positive])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([Anchor, Negative])

        # atten_pos_dist = Lambda(euclidean_distance, name='atten_pos_dist')([encoded_emb_atten, encoded_emb_atten_pos])
        # atten_neg_dist = Lambda(euclidean_distance, name='atten_neg_dist')([encoded_emb_atten, encoded_emb_atten_neg])
        # center_dist =  Lambda(euclidean_distance, name='center_dist')([encoded_emb, encoded_emb_atten])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg, emb_atten, emb_atten_pos, emb_atten_neg], stacked_dists, name='triple_siamese')
        import time
        model.summary()
        time.sleep(5.5)


        model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

        inter_layer = Model(inputs=model.get_input_at(0), outputs=model.get_layer('norm_layer').get_output_at(0))

        return model, inter_layer

    def load_triplets_model(self):
        model_dir = join(settings.OUT_DIR, 'model')
        rf = open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'r')
        model_json = rf.read()
        rf.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))
        return loaded_model

    def train_triplets_model(self):
        X1, X2, X3, X4, X5, X6 = self.load_triplets_data()
        n_triplets = len(X1)
        print('loaded')
        model, inter_model = self.create_triplet_model()
        # print(model.summary())

        X_anchor, X_pos, X_neg, X_atten, X_atten_pos, X_atten_neg = X1, X2, X3, X4, X5, X6
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg, 'attention_input': X_atten, 'attention_input_posive': X_atten_pos, 'attention_input_negive': X_atten_neg}
        model.fit(X, np.ones((n_triplets, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)

        model_json = model.to_json()
        model_dir = join(settings.OUT_DIR, 'model')
        os.makedirs(model_dir, exist_ok=True)
        with open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'w') as wf:
            wf.write(model_json)
        model.save_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))

        # don't need test
        test_triplets = self.load_triplets_data(role='test')
        auc_score = eval_utils.full_auc(model, test_triplets)
        # print('AUC', auc_score)

        loaded_model = self.load_triplets_model()
        print('triplets model loaded')
        auc_score = eval_utils.full_auc(loaded_model, test_triplets)

    def evaluate_triplet_model(self):
        test_triplets = self.load_triplets_data(role='test')
        loaded_model = self.load_triplets_model()
        print('triplets model loaded')
        auc_score = eval_utils.full_auc(loaded_model, test_triplets)


if __name__ == '__main__':
    global_model = GlobalTripletModel(data_scale=1000000)
    global_model.train_triplets_model()
    print('done')
    


import numpy as np


class HAN():
    def __init__(self):
        pass

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def load_data_dblp(self, truelabels, truefeatures, PAP, PSP, train_idx, val_idx, test_idx):
        rownetworks = [PAP, PSP]

        y = truelabels

        train_mask = self.sample_mask(train_idx, y.shape[0])
        val_mask = self.sample_mask(val_idx, y.shape[0])
        test_mask = self.sample_mask(test_idx, y.shape[0])

        y_train = np.zeros(y.shape)
        y_val = np.zeros(y.shape)
        y_test = np.zeros(y.shape)
        y_train[train_mask, :] = y[train_mask, :]
        y_val[val_mask, :] = y[val_mask, :]
        y_test[test_mask, :] = y[test_mask, :]

        # # return selected_idx, selected_idx_2
        # print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
        #                                                                                       y_val.shape,
        #                                                                                       y_test.shape,
        #                                                                                       train_idx.shape,
        #                                                                                       val_idx.shape,
        #                                                                                       test_idx.shape))
        truefeatures_list = [truefeatures, truefeatures, truefeatures]
        return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask



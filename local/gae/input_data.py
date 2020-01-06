from os.path import join
import numpy as np
import scipy.sparse as sp
from utils import settings
# from global_.prepare_local_data import IDF_THRESHOLD
IDF_THRESHOLD = 32

local_na_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(IDF_THRESHOLD))


def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))





if __name__ == '__main__':
    load_local_data(name='zhigang_zeng')

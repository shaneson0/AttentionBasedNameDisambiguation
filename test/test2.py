
from os.path import join
from utils import string_utils, settings
import json

authornames = ['Hongbin Li', 'Hua Bai', 'Kexin Xu', 'Lin Huang', 'Lu Han', 'Min Zheng', 'Qiang Shi', 'Rong Yu', 'Tao Deng', 'Wei Quan', 'Xu Xu', 'Yanqing Wang', 'Yong Tian']

Len = len(authornames)
for i in range(Len):
    authornames[i] = string_utils.clean_name(authornames[i])

with open(join(settings.DATA_DIR, 'test_name_list2.json'), 'w') as fp:
    json.dump(authornames, fp)
    fp.close()

# print(authornames)
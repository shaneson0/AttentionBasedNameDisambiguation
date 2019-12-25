
from HeterogeneousGraph.HAN import HAN
from utils import settings
from utils import data_utils, eval_utils
import codecs
from os.path import abspath, dirname, join
import numpy as np

def load_test_names():
    return data_utils.load_json(settings.DATA_DIR, 'test_name_list2.json')

# name="kexin_xu"
name = "hongbin_li"
# hongbin_li
# hongbin_li

han = HAN()

def testHAN(name):
    prec, rec, f1 = han.prepare_and_train(name=name)
    return prec, rec, f1
    # print (prec, rec, f1)

def main():
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name,precision,recall,f1\n')
    metrics = np.zeros(3)
    cnt = 0
    for name in names:

        prec, rec, f1 = han.prepare_and_train(name=name)
        print('name : ', name, 'prec, rec, f1: ', prec, rec, f1)
        # cur_metric, num_nodes, n_clusters = train(name, True)
        wf.write('{0},{1:.5f},{2:.5f},{3:.5f}\n'.format(
            name, prec, rec, f1))
        wf.flush()

        metrics[0] = metrics[0] + prec
        metrics[1] = metrics[1] + rec
        metrics[2] = metrics[2] + f1

        cnt += 1


    macro_prec = metrics[0] / cnt
    macro_rec = metrics[1] / cnt
    macro_f1 = eval_utils.cal_f1(macro_prec, macro_rec)
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1))
    wf.close()

if __name__ == '__main__':
    # main()
    # testHAN(name)
    names = load_test_names()
    wf = codecs.open(join(settings.OUT_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
    wf.write('name,precision,recall,f1\n')
    macro_prec, macro_rec, macro_f1, count = 0.0, 0.0, 0.0, 0.0
    for name in names:

        # prec, rec, f1 = han.prepare_and_train(name=name)
        prec, rec, f1 = testHAN(name)
        print('name : ', name, 'prec, rec, f1: ', prec, rec, f1)
        # cur_metric, num_nodes, n_clusters = train(name, True)
        wf.write('{0},{1:.5f},{2:.5f},{3:.5f}\n'.format(
            name, prec, rec, f1))
        wf.flush()
        macro_prec += prec
        macro_rec += rec
        macro_f1 += f1
        count += 1

        break

    macro_prec /= count
    macro_rec /= count
    macro_f1 /= count

    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        macro_prec, macro_rec, macro_f1))
    wf.close()







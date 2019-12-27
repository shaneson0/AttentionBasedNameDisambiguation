from os.path import join
import os
from utils import data_utils
from utils import settings

class DataGenerator:
    pids_train = []
    def __init__(self):
        pass


    def prepare_data(self):
        self.name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # for test
        self.name2pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
        print ("self.name2pubs_train :", self.name2pubs_train )
        print ("self.name2pubs_test: ", self.name2pubs_test )
        self.names_train = self.name2pubs_train.keys()
        self.names_test = self.name2pubs_test.keys()
        assert not set(self.names_train).intersection(set(self.names_test))

        for name in self.names_train:
            name_pubs_dict = self.name2pubs_train[name]
            for aid in name_pubs_dict:
                self.pids_train += name_pubs_dict[aid]




    def run(self):
        self.prepare_data()


if __name__ == '__main__':
    datagenerator = DataGenerator()
    datagenerator.run()
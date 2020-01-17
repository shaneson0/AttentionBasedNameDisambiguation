
from utils import data_utils, settings


name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')

# Train
TrainAuthorCount = 0
TrainPaperCount = 0
for name in name_to_pubs_train:
    TrainAuthorCount = TrainAuthorCount + 1
    object = name_to_pubs_train[name]
    for pid in  object:
        TrainPaperCount = TrainPaperCount + 1

print ("TrainAuthorCount: ", TrainAuthorCount)
print ("TrainPaperCount: ", TrainPaperCount)

# Test
TestAuthorCount = 0
TestPaperCount = 0
for name in name_to_pubs_test:
    TestAuthorCount = TestAuthorCount + 1
    object = name_to_pubs_test[name]
    for pid in object:
        TestPaperCount = TestPaperCount + 1

print ("TestAuthorCount: ", TestAuthorCount)
print ("TestPaperCount: ", TestPaperCount)



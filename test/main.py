import codecs
from os.path import join
from utils import settings


def loadAuthorSocial():
    AuthorSocial = {}
    with codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_social.txt'), 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            items = line.rstrip().split('\t')
            print(items[0], ' ', list(map(lambda x: int(x) , items[1].split(' '))))
            AuthorSocial[items[0]] = list(map(lambda x: int(x) , items[1].split(' ')))

loadAuthorSocial()


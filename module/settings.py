ROOT = '/Users/xinai/github/toxic-comment-classification/'
DATA = ROOT + 'data/'
MODEL = ROOT + 'model/'

ORIGINAL = DATA + 'train.csv'
TRAIN = DATA + 'my_train'
TEST = DATA + 'my_test'

TFIDF = MODEL + 'tfidf'

TEXT_COLUMN = 'comment_text'
DEFAULT_SCORE_COLUMN = 'toxic'

import settings
import utils
import data_utils
import text_utils

import sys
import time

from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Loading test data')
sys.stdout.flush()
x_raw, y = data_utils.load_data(settings.TEST, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

print('Generating features')
sys.stdout.flush()
start = time.time()
tfidf = joblib.load()
x_feat = tfidf.fit_transform(x_raw)
end = time.time()
print('Total time spent: {}'.format(utils.seconds_to_str(end - start)))

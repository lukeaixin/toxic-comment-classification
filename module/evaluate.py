import settings
import utils
import data_utils
import text_utils

from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Loading test data'):
x_raw, y = data_utils.load_data(settings.TEST, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

with utils.log_message('Generating features'):
    tfidf = joblib.load(settings.TFIDF)
    x_feat = tfidf.transform(x_raw)

utils.log_message('Complete\n')

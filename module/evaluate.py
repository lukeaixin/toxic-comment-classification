import settings
import utils
import data_utils
import text_utils

from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Starting script to evaluate on test set')

print('Loading test data and models')
x_raw, y = data_utils.load_data(settings.TEST, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)
tfidf = joblib.load(settings.TFIDF)
lr = joblib.load(settings.LR_TFIDF)

with utils.log_runtime('Generating features'):
    x_feat = tfidf.transform(x_raw)

print('Predicting and evaluating')
acc = lr.score(x_feat, y)
idx = (lr.classes_ == 1)
prob = lr.predict_proba(x_feat)[:, idx]
auc = roc_auc_score(y, prob)
print('Test Accuracy: {}'.format(acc))
print('Test AUC: {}'.format(auc))

print('Complete\n')

import settings
import utils
import data_utils
import text_utils

import sys
import time

from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Loading training data')
sys.stdout.flush()
_, y = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

print('Loading features')
x_feat = load_npz(settings.TFIDF_FEAT)

print('Training model')
sys.stdout.flush()
start = end
lr = SGDClassifier(loss='log', alpha=1e-4, max_iter=1000, tol=1e-3, class_weight='balanced')
lr.fit(x_feat, y)
end = time.time()
print('Total time spent: {}'.format(utils.seconds_to_str(end - start)))

idx = (lr.classes_ == 1)

print('Training Accuracy: {}'.format(lr.score(x_feat, y)))
print('Training AUC: {}'.format(roc_auc_score(y, lr.predict_proba(x_feat)[:, idx])))

print('Saving models')
sys.stdout.flush()
joblib.dump(lr, settings.LR_TFIDF)

print('Complete')

import settings
import utils
import data_utils
import text_utils

from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Loading training data')
_, y = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

print('Loading features')
x_feat = load_npz(settings.TFIDF_FEAT)

with utils.log_runtime('Training model'):
    lr = SGDClassifier(loss='log', alpha=1e-4, max_iter=1000, tol=1e-3, class_weight='balanced')
    lr.fit(x_feat, y)

idx = (lr.classes_ == 1)
print('Training Accuracy: {}'.format(lr.score(x_feat, y)))
print('Training AUC: {}'.format(roc_auc_score(y, lr.predict_proba(x_feat)[:, idx])))

print('Saving models'):
joblib.dump(lr, settings.LR_TFIDF)

print('Complete\n')

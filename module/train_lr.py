import settings
import utils
import data_utils

from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Starting script to train lr model')

print('Loading training data')
_, y = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)
x_feat = load_npz(settings.TFIDF_FEAT)

with utils.log_runtime('Training model'):
    lr = SGDClassifier(loss='log', alpha=1e-4, max_iter=1000, tol=1e-3, class_weight='balanced')
    lr.fit(x_feat, y)
    acc = lr.score(x_feat, y)
    idx = (lr.classes_ == 1)
    prob = lr.predict_proba(x_feat)[:, idx]
    auc = roc_auc_score(y, prob)
    print('Training Accuracy: {}'.format(acc))
    print('Training AUC: {}'.format(auc))

print('Saving model')
joblib.dump(lr, settings.LR_TFIDF)

print('Complete\n')

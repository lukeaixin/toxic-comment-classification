import settings
import data_utils
import text_utils

import sys
import time
from datetime import timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

print('Loading training data')
sys.stdout.flush()
x_raw, y = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

print('Generating features')
sys.stdout.flush()
start = time.time()
tfidf = TfidfVectorizer(preprocessor=text_utils.prep_text, max_df=0.95, max_features=5000)
x_feat = tfidf.fit_transform(x_raw)
end = time.time()
print('Total time spent: ' + str(timedelta(end - start)))


print('Training model')
sys.stdout.flush()
start = end
alpha = float(sys.argv[1])
lr = SGDClassifier(loss='log', alpha=alpha, max_iter=1000, tol=1e-3, class_weight='balanced')
lr.fit(x_feat, y)
end = time.time()
print('Total time spent: ' + str(timedelta(end - start)))

idx = (lr.classes_ == 1)

print('Training Accuracy: {}'.format(lr.score(x_feat, y)))
print('Training AUC: {}'.format(roc_auc_score(y, lr.predict_proba(x_feat)[:, idx])))

print('Saving models')
sys.stdout.flush()
joblib.dump(tfidf, settings.TFIDF)
joblib.dump(lr, settings.LR_TFIDF)

print('Complete')

import settings
import utils
import data_utils
import text_utils

import sys
import time

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

print('Loading training data')
sys.stdout.flush()
x_raw, _ = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

print('Generating features')
sys.stdout.flush()
start = time.time()
tfidf = TfidfVectorizer(preprocessor=text_utils.prep_text, max_df=0.95, max_features=5000)
x_feat = tfidf.fit_transform(x_raw)
end = time.time()
print('Total time spent: {}'.format(utils.seconds_to_str(end - start)))

print('Saving models')
sys.stdout.flush()
joblib.dump(tfidf, settings.TFIDF)

print('Saving features')
sys.stdout.flush()
save_npz(settings.TFIDF_FEAT, x_feat)

print('Complete')

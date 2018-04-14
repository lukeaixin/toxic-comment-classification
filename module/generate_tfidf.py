import settings
import utils
import data_utils
import text_utils

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

print('Loading training data')
x_raw, _ = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

with utils.log_runtime('Generating features'):
    tfidf = TfidfVectorizer(preprocessor=text_utils.prep_text, max_df=0.95, max_features=5000)
    x_feat = tfidf.fit_transform(x_raw)

print('Saving models')
joblib.dump(tfidf, settings.TFIDF)

print('Saving features')
save_npz(settings.TFIDF_FEAT, x_feat)

print('Complete\n')

import settings
import utils
import data_utils
import text_utils

import os

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

if os.path.exists(settings.TFIDF) and os.path.exists(settings.TFIDF_FEAT):
    print('Skipping script to generate tfidf')
    print('Model and features already exists\n')
else:
    print('Starting script to generate tfidf')

    print('Loading data')
    x_raw, _ = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)

    with utils.log_runtime('Fitting model and generating features'):
        tfidf = TfidfVectorizer(preprocessor=text_utils.prep_text, max_df=0.95, max_features=5000)
        x_feat = tfidf.fit_transform(x_raw)

    print('Saving model and features')
    joblib.dump(tfidf, settings.TFIDF)
    save_npz(settings.TFIDF_FEAT, x_feat)

    print('Complete\n')

import settings
import data_utils
import text_utils

from sklearn.feature_extraction.text import TfidfVectorizer

x_raw, _ = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)
tfidf = TfidfVectorizer(preprocessor=text_utils.prep_text, max_df=0.95, max_features=5000)
feat = tfidf.fit_transform(x_raw)



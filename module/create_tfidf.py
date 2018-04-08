import sys

from sklearn.feature_extraction.text import TfidfVectorizer

import settings
import data_utils
import text_utils

x, _ = data_utils.load_data(settings.TRAIN, settings.TEXT_COLUMN, settings.DEFAULT_SCORE_COLUMN)
x = x[:5]

for t in x:
    print '\n\n\nTHIS IS A COMMENT\n\n\n'
    print t
    print text_utils.prep_text(t)
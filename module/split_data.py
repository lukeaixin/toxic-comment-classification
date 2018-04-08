import settings

import pandas as pd
from sklearn.model_selection import train_test_split

ori_file = settings.ORIGINAL
train_file = settings.TRAIN
test_file = settings.TEST

ori = pd.read_csv(ori_file, header=0)

train, test = train_test_split(ori, test_size=0.3)

train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

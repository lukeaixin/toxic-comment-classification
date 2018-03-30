import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ori_file = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]

ori = pd.read_csv(ori_file, header=0)

train, test = train_test_split(ori, test_size=0.3)

train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

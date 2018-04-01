import numpy as np
import pandas as pd

def load_data(file, x_columns, y_column):
    data = pd.read_csv(file, header=0)
    return np.array(data.ix[:, x_columns]), np.array(data.ix[:, y_column])

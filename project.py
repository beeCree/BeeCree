import numpy as np
import pandas as pd
import gc
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")
application_train = pd.read_csv('data_train.csv')
catFeatures = [4, 5, 6, 7, 13, 17]

import numpy as np
import pandas as pd
import gc
import warnings
import time
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


def cutConcat(array1,col_index): ## given ndarray and int, returns ndarray
    return np.concatenate((array1[:,:col_index],array1[:,col_index+1:]),axis=1)
def length(array1): ## given ndarray, returns int
    return array1.shape[0]
def width(array1): ## given ndarray, returns int
    return array1.shape[1]

def dateColumn2Float(dataframe, col_index): ## given dataframe and int, returns dataframe
    array1 = dataframe.values
    for i in range(length(array1)):
        date = pd.to_datetime(array1[i,col_index].replace("-",""), format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
        day_of_year = (date - new_year_day).days + 1
        array1[i, col_index] = day_of_year/365 + date.year
    return pd.DataFrame(array1)
        
print(cutConcat(np.array([[1,2,3],[4,5,6]]),1))
warnings.filterwarnings("ignore")
application_train = pd.read_csv('test.csv') ##data_train.csv이어야 하지만 테스트용
## input data에서 첫 줄을 띄어줘야 한다.
## ***********************************************************
## train file에서 feature과 price를 자른다.
y=application_train[["price"]]
application_train.drop("price", axis=1,inplace=True)
print("y=")
print(y)
## *********************************************************
cat_features = [4, 5, 6, 7, 13, 17] ## 실제 데이터에서 카테고리 데이터의 인덱스
non_cat_features = []
for i in range(application_train.shape[1]):
    if i in cat_features:
        pass
    else:
        non_cat_features.append(i)
## 카테고리 데이터와 아닌 데이터들의 인덱스 분류
## ********************************************************
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
application_train=application_train.replace(float('nan'), -123)
onehot_encoded = onehot_encoder.fit_transform(application_train.iloc[:,cat_features]).toarray()
print("Executing One-Hot encoder...")
print(onehot_encoded)
non_cat_data = application_train.iloc[:,non_cat_features]
total_data_nd = np.concatenate((non_cat_data, onehot_encoded), axis=1)
total_data_df = pd.DataFrame(total_data_nd).replace(-123,float('nan'))

total_data_df = dateColumn2Float(total_data_df, 0)##0번째 feature은 날짜임

total_data_df = dateColumn2Float(total_data_df, 12)
##원래 18번째 feature가 날짜지만 카테고리데이터가 뒤로 가서 12번째로 당겨짐
##항상 12번째로 오므로 딱히 문제는 없다고 생각

print("Converting data to float and reproducing NaNs...")
print(total_data_df)
## one hot encoding finish
## ************************************************************
"""
total_data_df = Imputer(strategy='median').fit_transform(total_data_df)
print("Imputing...")
print(total_data_df)
"""##imputer 부분은 일단 주석처리함

##**************************************************************
## random forest로 feature selection하기
clf = RandomForestClassifier(n_estimators = 1000, random_state = 0, n_jobs = -1)
clf.fit(total_data_df,y)

## 각 feature별 중요도 출력
for i in range(len(clf.feature_importances_)):
    print('{} = {}'.format(i,clf.feature_importances_[i]))

## 중요도가 0.03이상인 feature를 select하여 list를 반환
sfm = SelectFromModel(clf, threshold=0.03)
sfm.fit(total_data_df,y)
print("sfm=")
## 이 함수를 쓰면 feature의 index를 담은 list가 나옴
print(sfm.get_support(indices = True))

## 중요도가 높은 feature를 걸러낸 data
for index in range(len(clf.feature_importances_)):
    if not index in sfm.get_support(indices = True):
        pd.DataFrame(total_data_df).drop(index, axis=1,inplace=True)
print("total_data_df = ")
print(total_data_df)
##**************************************************************



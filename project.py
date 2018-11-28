import numpy as np
import pandas as pd
import gc
import warnings
import time
import sys
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from math import isnan
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
FILE_NAME =1
MAKEY_SKIP = 2
ONEHOT_SKIP = 3
IMPUTE_SKIP = 4
KIND_OF_FS = 5
##**************************************************************************************
## 1. declaritive coding style로 더 보기 편하게 만들었음
## 2. concat같이 오래 걸리는 한 번만 해도 되는 작업은
##  command line argument로 skip여부를 확인하게 바꾸었음.
##  형식;
##  project.py
##      (file name) (skip makey?) (skip onehot과 전처리?) (skip impute?) (name of fs)
##  skip하고 싶다면 대문자 "Y"를 입력시 skip됨.
##  name of fs는 당연히 쓸 feature selection의 이름을 입력하거나,
##  all을 입력하면 implement된 모든 feature selection을 다 execute함.
##  물론 당연히 skip을 하려면 그 파일에 대해서 한 번은 미리 실행을 해 보아야 함.
##  위에 const로 정의해놓았으니 입력 방식을 바꾸고 싶으면
##  import 아래를 참조.
##**************************************************************************************
def cutConcat(array1,col_index): ## given ndarray and int, returns ndarray
    return np.concatenate((array1[:,:col_index],array1[:,col_index+1:]),axis=1)
def length(array1): ## given ndarray, returns int
    return array1.shape[0]
def width(array1): ## given ndarray, returns int
    return array1.shape[1]
def dateColumn2Float(dataframe, col_index): ## given dataframe and int, returns dataframe
    array1 = dataframe.values
    for i in range(length(array1)):
        if (type(array1[i, col_index]) is float):
            if (isnan(array1[i, col_index])):
                pass
        else:
            ##print(array1[i,col_index])
            date = pd.to_datetime(array1[i,col_index].replace("-",""), format='%Y%m%d')
            new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
            day_of_year = (date - new_year_day).days + 1
            array1[i, col_index] = day_of_year/365 + date.year
    return pd.DataFrame(array1)
def makey(application_train):
    print("Making y...")
    if(sys.argv[MAKEY_SKIP] == "Y"):
        y = pd.read_csv('y.csv')
        print("y=")
        print(y)
        gc.collect()
        return y
    else:
        y = application_train[["price"]]
        application_train.drop("price", axis=1, inplace=True)
        print("y=")
        print(y)
        y.to_csv("y.csv", index=False)
        gc.collect()
        return y
def onehotNd2f(application_train, y): ## input이 "skip"이면 skip. else not skip
    print("Reading data_train.csv and One-Hot encoder and imputing...")
    if(sys.argv[ONEHOT_SKIP] == "Y"):
        gc.collect()
        total_data_df = pd.read_csv('concat.csv')
        return total_data_df
    else:
        gc.collect()
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
        print("Executing One-Hot encoder...")
        onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        application_train=application_train.replace(float('nan'), -123)
        onehot_encoded = onehot_encoder.fit_transform(application_train.iloc[:,cat_features]).toarray()
        print("done.")
        print(onehot_encoded)
        
        print("Concatenating Array...")
        non_cat_data = application_train.iloc[:,non_cat_features]
        total_data_nd = np.concatenate((non_cat_data, onehot_encoded), axis=1)
        total_data_df = pd.DataFrame(total_data_nd).replace(-123,float('nan'))
        
        print("done.\n")
        print("Converting data to float and reproducing NaNs...")
        total_data_df = dateColumn2Float(total_data_df, 0) ##0번째 feature은 날짜.
        
        total_data_df = dateColumn2Float(total_data_df, 12)
        ##원래 18번째 feature가 날짜지만 카테고리데이터가 뒤로 가서 12번째로 당겨짐
        ##항상 12번째로 오므로 딱히 문제는 없다고 생각
        print("done.")
        print(total_data_df)
        print("Saving data on concat.csv")
        total_data_df.to_csv("concat.csv", index=False)
        return total_data_df
        ## one hot encoding finish
        ## *************************************************************
def impute(total_data_df):
    print("Imputing...")
    if(sys.argv[IMPUTE_SKIP] == "Y"):
        gc.collect()
        total_data_df = pd.read_csv('impute.csv')
        return total_data_df
    else:
        gc.collect()
        total_data_df = Imputer(strategy='median').fit_transform(total_data_df)
        print("done.")
        print(total_data_df)
        print("Saving data on impute.csv")
        pd.DataFrame(total_data_df).to_csv("impute.csv", index=False)
        return total_data_df
def selectfs(total_data_df):
    if(sys.argv[KIND_OF_FS] == "wrapper" or sys.argv[KIND_OF_FS] == "all"):
        wrapper(total_data_df)
    if(True):
        pass
def wrapper(total_data_df):
    print("Wrapping...")
    data_norm = MinMaxScaler().fit_transform(total_data_df)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=30, step=10, verbose=5)
    rfe_selector.fit(data_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = pd.DataFrame(total_data_df).loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    return rfe_feature
    
gc.collect()        
warnings.filterwarnings("ignore")
## input data에서 첫 줄을 띄어줘야 한다.
## train file에서 feature과 price를 자른다.

application_train = pd.read_csv(sys.argv[FILE_NAME]) ##data_train.csv이어야 하지만 테스트용

y=makey(application_train)

total_data_df = onehotNd2f(application_train, y)
gc.collect()

total_data_df = impute(total_data_df)
gc.collect()

features = selectfs(total_data_df)
gc.collect()

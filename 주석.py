import numpy as np
import pandas as pd
import gc
import warnings
import sys
import os
import math
from math import isnan
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
FILE_NAME =1 ## input file 이름
MAKEY_SKIP = 2 ## makey procedure가 이미 실행 된 경우 skip할 지 결정
KIND_OF_ENC = 3 ## 어떤 encoder를 사용할지 결정(only non used)
ENCODER_SKIP = 4 ## encoder가 이미 실행 된 경우 skip할 지 결정
IMPUTE_SKIP = 5 ## impute이 이미 실행 된 경우 skip할 지 결정
M_FEATURES = 20
CONST_DATE_INDEX = 13

def log_y(y): ##y 행렬 원소들 각각 log
    logy = np.zeros((length(y),1))
    for i in range(length(y)):
        logy[i,0] = math.log(y[i])*10000
    return pd.DataFrame(logy)

def exp_y(y): ##y 행렬 원소들 각각 exp 
    expy = np.zeros((length(y),1))
    for i in range(length(y)):
        expy[i,0] = math.exp(y[i]/10000)
    return expy

def sindeg(x): ## deg로 값을 input받는 sin 함수.
    return math.sin(x*math.pi/180)
def cosdeg(x): ## deg로 값을 input받는 cos 함수.
    return math.cos(x*math.pi/180)

def handle_deg(application_train):
    ## given data의 각도 값을 sin과 cos으로 바꾸어 각도가 연속적으로 변화하게 만든다.
    deg = application_train[["9"]]
    deg1 = deg.values
    sincos = np.zeros((length(deg1),2))
    for i in range(length(deg1)):
        sincos[i,0] = sindeg(deg1[i])
        sincos[i,1] = cosdeg(deg1[i])
    array1 = application_train.values
    temp1 =  np.concatenate((array1[:,:9],sincos),axis=1)
    result = np.concatenate((temp1,array1[:,10:]),axis=1)
    return pd.DataFrame(result)

def cutConcat(array1,col_index): ## given ndarray and int, returns ndarray
    return np.concatenate((array1[:,:col_index],array1[:,col_index+1:]),axis=1)
def length(array1): ## given ndarray, returns the length of array
    return array1.shape[0]
def width(array1): ## given ndarray, returns the width of array
    return array1.shape[1]
def filename(): ## 경로 string "FILE_NAME\KIND_OF_ENC\"를 만든다.
    return sys.argv[FILE_NAME][:-4]+'/'+sys.argv[KIND_OF_ENC]
def dateColumn2Float(dataframe, col_index):
    ## 날짜 string은 일반적인 계산이 불가능하므로, float로 변환하여(1일 = 1/365년)
    ## float처럼 계산할 수 있게 만든다.
    array1 = dataframe.values
    for i in range(length(array1)):
        if (type(array1[i, col_index]) is float):
            if (isnan(array1[i, col_index])):
                pass
        else:
            date = pd.to_datetime(array1[i,col_index].replace("-",""), format='%Y%m%d')
            new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
            day_of_year = (date - new_year_day).days + 1
            array1[i, col_index] = day_of_year/365 + date.year
    return pd.DataFrame(array1)
def makey(application_train):
    ## 결과값을 데이터에서 미리 분리해 둬서 return하는 함수이다.
    print("Making y...")
    if(sys.argv[MAKEY_SKIP] == "Y"):
        y = pd.read_csv(filename()+'/y.csv')
        print("y=")
        print(y)
        gc.collect()
        return y
    else:
        y = application_train[["price"]]
        application_train.drop("price", axis=1, inplace=True)
        print("y=")
        print(y)
        y.to_csv(filename()+'/y.csv', index=False)
        gc.collect()
        return y

def enc_selector(application_train, y):
    ## categorical data를 handle하기 위해서, 어떤 encoder를 사용할 지 선택하는 함수였지만,
    ## 이를 모델에 적용시켰을 때, train error가 비슷했기 때문에 그냥 dimension이 가장 작아서
    ## overfitting이 적게 encoder를 쓰지 않게 했다.
    if(sys.argv[KIND_OF_ENC] == "bin"):
        print("Binary encoder selected.")
        return binaryNd2f(application_train, y)
    if(sys.argv[KIND_OF_ENC] == "ohc"):
        print("One hot encoder selected.")
        return onehotNd2f(application_train, y)
    if(sys.argv[KIND_OF_ENC] == "non"):
        print("Not encoding selected")
        return notEncode(application_train,y)
        
def binaryNd2f(application_train, y):
    ## unused
    ## categorical data의 value를 이진수로 나타내서 각 자리를 한 feature로 나타내어 encode했다.
    if(sys.argv[ENCODER_SKIP] == "Y"):
        print("Skipping binary encoder")
        gc.collect()
        total_data_df = pd.read_csv(filename()+'/concat_b.csv')
        return total_data_df
    print("Executing binary encoder..")
    ## handle degree
    application_train = handle_deg(application_train)
    gc.collect()
    cat_features = [4, 5, 6, 7, 14, 18]
    non_cat_features = []
    for i in range(application_train.shape[1]):
        if i in cat_features:
            pass
        else:
            non_cat_features.append(i)
    cat_max = []
    cat_min = []
    dims = []
    for i in cat_features:
        temp_max = -123456
        temp_min = 123456
        for j in range(length(application_train)):
            if(np.isnan(application_train.iloc[j, i])):
                pass
            else:
                if(temp_max < application_train.iloc[j, i]):
                    temp_max = application_train.iloc[j, i]
                if(temp_min > application_train.iloc[j, i]):
                    temp_min = application_train.iloc[j, i]
        cat_max.append(temp_max)
        cat_min.append(temp_min)
        dims.append(int(math.ceil(np.log2(temp_max-temp_min+1))))
    binenc_info = np.concatenate((np.array(cat_min), np.array(dims)))
    pd.DataFrame(binenc_info).to_csv(filename()+'/binenc_info.csv', index=False)
    non_cat_train = application_train.iloc[:,non_cat_features].values
    for i in range(6):
        binencoded = np.zeros((length(application_train),dims[i]))
        for j in range(length(application_train)):
            temp = application_train.iloc[j, cat_features[i]]-cat_min[i]
            for k in range(int(math.ceil(dims[i]))):
                binencoded[j, k] = temp%2
                if(np.isnan(temp)):
                    pass
                else:
                    temp = math.floor(temp/2)
        non_cat_train = np.concatenate((non_cat_train, binencoded), axis=1)
    bin_concated=pd.DataFrame(non_cat_train)
    bin_concated = dateColumn2Float(bin_concated, 0)        
    bin_concated = dateColumn2Float(bin_concated, CONST_DATE_INDEX)
    print("done.")
    bin_concated.to_csv(filename()+'/concat_b.csv', index=False)
    print(bin_concated)
    return bin_concated

def onehotNd2f(application_train, y): 
    if(sys.argv[ENCODER_SKIP] == "Y"): ## input이 "skip"이면 skip. else not skip
        gc.collect()
        total_data_df = pd.read_csv(filename()+'/concat_o.csv')
        return total_data_df
    else:
        gc.collect()
        ## handle degree
        application_train = handle_deg(application_train)
        ## *********************************************************
        cat_features = [4, 5, 6, 7, 14, 18] ## 실제 데이터에서 카테고리 데이터의 인덱스
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
        
        total_data_df = dateColumn2Float(total_data_df, CONST_DATE_INDEX)
        ##원래 18번째 feature가 날짜지만 카테고리데이터가 뒤로 가서 12번째로 당겨짐
        ##항상 12번째로 오므로 딱히 문제는 없다고 생각
        print("done.")
        print(total_data_df)
        print("Saving data on concat_o.csv")
        total_data_df.to_csv(filename()+'/concat_o.csv', index=False)
        return total_data_df
        ## one hot encoding finish
        ## *************************************************************

def notEncode(application_train, y):
    ## encoding을 하지 않고, given data의 날짜를 float로 고치기만 하는 함수.
    if(sys.argv[ENCODER_SKIP] == "Y"):
        print("Skipping binary encoder")
        gc.collect()
        total_data_df = pd.read_csv(filename()+'/concat_n.csv')
        return total_data_df
    print("Not encoder..")
    ## handle degree
    application_train = handle_deg(application_train)
    gc.collect()
    application_train=dateColumn2Float(application_train, 0) ##0번째 feature은 날짜.
    application_train = dateColumn2Float(application_train, CONST_DATE_INDEX + 6)
    print("done")
    application_train = pd.DataFrame(application_train)
    ## 완료된 결과를 filename()+'/concat_n.csv'에 저장한다.
    application_train.to_csv(filename()+'/concat_n.csv', index=False)
    return application_train

def impute(total_data_df):
    ## sklearn의 Imputer함수를 이용해서, 빈 칸을 채우는 함수.
    ## 중간값을 빈 칸에 채운다.
    print("Imputing...")
    if(sys.argv[IMPUTE_SKIP] == "Y"):
        gc.collect()
        total_data_df = pd.read_csv(filename()+'/impute.csv')
        return total_data_df
    else:
        gc.collect()
        total_data_df = Imputer(strategy='median').fit_transform(total_data_df)
        print("done.")
        print(total_data_df)
        print("Saving data on impute.csv")
        pd.DataFrame(total_data_df).to_csv(filename()+'/impute.csv', index=False)
        return total_data_df

def gbr(total_data_df, y, testx, testy):
    ## make a model using gradient boosting regressor tree
    print("gradientboostingregressor")
    ## maximum depth of tree is 9
    GBR = GradientBoostingRegressor(n_estimators = 100, max_depth = 5)
    GBR.fit(total_data_df,log_y(y))
    acc=accuracy(testy,exp_y(GBR.predict(testx)))
    print("Accuracy --> ", acc)
    return acc

def accuracy(real, pred):
    ## calculate accuracy between real data and predicted data
    ## accuracy is calculated as same as slide
    total = 0
    for i in range(len(real)):
        total=total+abs((real[i]-pred[i])/real[i])
    return 1-1/len(real)*total
def fold5(total_data_df, y):
    ## beautiful code implementing 5-fold cross vallidation
    X = total_data_df.values
    ynd = y.values
    dim_mody = width(X)
    list0=[]
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    for i in range(len(X)):
        if(i%5==0):
            list0.append(i)
        if(i%5==1):
            list1.append(i)
        if(i%5==2):
            list2.append(i)
        if(i%5==3):
            list3.append(i)
        if(i%5==4):
            list4.append(i)
    listn4 =sorted(list0 + list1 + list2 + list3)
    listn3 =sorted(list0 + list1 + list2 + list4)
    listn2 =sorted(list0 + list1 + list3 + list4)
    listn1 =sorted(list0 + list2 + list4 + list3)
    listn0 =sorted(list1 + list2 + list3 + list4)
    print("yndn0 = ", ynd[listn0,:])
    print("yndn1 = ", ynd[listn1,:])
    print("yndn2 = ", ynd[listn2,:])
    print("yndn3 = ", ynd[listn3,:])
    print("yndn4 = ", ynd[listn4,:])
    acc0 = gbr(X[listn0,:], ynd[listn0,:],X[list0,:],ynd[list0,:])
    acc1 = gbr(X[listn1,:], ynd[listn1,:],X[list1,:],ynd[list1,:])
    acc2 = gbr(X[listn2,:], ynd[listn2,:],X[list2,:],ynd[list2,:])
    acc3 = gbr(X[listn3,:], ynd[listn3,:],X[list3,:],ynd[list3,:])
    acc4 = gbr(X[listn4,:], ynd[listn4,:],X[list4,:],ynd[list4,:])
    print((acc0+acc1+acc2+acc3+acc4)/5)
if(not os.path.exists(filename())):
    os.makedirs(filename())
gc.collect()        
warnings.filterwarnings("ignore")

application_train = pd.read_csv(sys.argv[FILE_NAME])
y=makey(application_train)

total_data_df = enc_selector(application_train, y)
gc.collect()

total_data_df = pd.DataFrame(impute(total_data_df))
gc.collect()
fold5(total_data_df, y)

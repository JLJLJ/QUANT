import QUANTAXIS as QA
import pandas as pd
import numpy as np

import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functools import wraps
import time
from joblib import Parallel, delayed
import multiprocessing

print('load jp')
def fn_timer(function):
  @wraps(function)
  def function_timer(*args, **kwargs):
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    print ("Total time running %s: %s seconds" %
        (function.__name__, str(t1-t0)))
    return result
  return function_timer

def func_scale(df,groups):
    #部分不同列的数据具有相关性，需要统一标准化
    for group in groups:
        if type(group)!=list and type(group)!=tuple:
            group=[group]
        assemble_scale_column=group
        ar=np.ravel(df[assemble_scale_column]).reshape([len(df),-1])
        ar=ar/ar.mean()
        df[assemble_scale_column] = preprocessing.scale(ar)
    return df

def standardization(df,groups):
    #对数据去中心化，使符合正态分布
    df=df.groupby('code').apply(func_scale,groups)
    return df


#时间序列转换为监督学习样本
def series_to_supervised(dataset, n_in=1):
    df_all=pd.DataFrame([])
    for i in range(n_in, -1, -1):      
        df_back=dataset.shift(i)      #把数据整体下移i格
        col_names=df_back.columns+'_'+str(i)   #数字后缀代表距当前的窗口期期数
        df_back.columns=col_names
        df_all=pd.concat([df_all,df_back],axis=1)
        #print(df_all.index)
    return df_all

def series_to_supervised_parallel(data, n_in=1):
    re=Parallel(n_jobs=40)(delayed(series_to_supervised)(df,n_in) for code,df in data.groupby('code'))
    re=pd.concat(re)
    return re


#使用单条样本内相关分组进行标准化（去均值化）
def func_scale_axis1(se,groups):
    #部分不同列的数据具有相关性，需要统一标准化
    col_names=se.index
    for group in groups:
        col=[]
        group=pd.Series(group)
        group='^'+group+'_\d+$'
        ind=se.index
        for g in group:
            col=col+ind[ind.str.match(g)].to_list()
        assemble_scale_column=col
        ar=np.ravel(se[assemble_scale_column])
        ar=ar/ar.mean()    #去均值的中心化（均值变为0）
        se[assemble_scale_column] = ar
    return se

def standardization_axis1(dataset,col_groups):
    dataset=dataset.copy()
    df=dataset.apply(func_scale_axis1, args=(col_groups,),axis=1)
    return df

def standardization_axis1_parallel(data,col_groups):
    re=Parallel(n_jobs=40)(delayed(standardization_axis1)(df, col_groups) for code,df in data.groupby('code'))
    re=pd.concat(re)
    return re

def drop_columns(df,cols):
    #部分不同列的数据具有相关性，需要统一标准化
    col_names=[]
    for col in cols:
        col='^'+col+'_\d+$'
        col_names=col_names+df.columns[df.columns.str.match(col)].to_list()
        df=df.drop(col_names,axis=1,errors='ignore')
    return df



def max_min_close_n_days(data, n=30):
    '''
    自定义指标，计算当日（不含）到n天后收盘价最大值、最小值、最大值上涨比率、最小值下跌比率
    '''
    n_days_max=pd.Series(np.zeros(len(data)))
    n_days_min=pd.Series(np.zeros(len(data)))
    n_days_max_radio=pd.Series(np.zeros(len(data)))
    n_days_min_radio=pd.Series(np.zeros(len(data)))
    if(len(data)-n<0):
        '''for i in range(len(data)):
            n_days_max[i]=None
            n_days_min[i]=None'''
        n_days_max[0:len(data)]=None
        n_days_min[0:len(data)]=None
    else:
        for i in range(len(data)-n):
            n_days_max[i]=(max(data['close'][i+1:i+n+1]))
            n_days_min[i]=(min(data['close'][i+1:i+n+1]))
            n_days_max_radio[i]=(n_days_max[i]-data['close'][i])/data['close'][i]
            n_days_min_radio[i]=(n_days_min[i]-data['close'][i])/data['close'][i]
        n_days_max[len(data)-n:len(data)]=None
        n_days_min[len(data)-n:len(data)]=None
        '''for i in range(len(data)-n,len(data)):
            n_days_max[i]=None
            n_days_min[i]=None
            '''
    max_min_close_n_days=pd.DataFrame({'n_days_max':n_days_max,'n_days_min':n_days_min,
                                       'n_days_max_radio':n_days_max_radio,'n_days_min_radio':n_days_min_radio})
    max_min_close_n_days.index=data.index
    return max_min_close_n_days

def get_all_indicator(data):
    dataset=pd.DataFrame()

    indicator=QA.QA_indicator_ADTM(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_ARBR(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_ASI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_ATR(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_BBI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    #indicator=QA.QA_indicator_BIAS(data)
    #dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_BOLL(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_CCI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_CHO(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_DDI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_DMA(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_DMI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    #indicator=QA.QA_indicator_EMA(data)
    #dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_EXPMA(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_KDJ(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_MA(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_MA_VOL(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_MACD(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_MFI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_MIKE(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_MTM(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_OBV(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_OSC(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_PBX(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_PVT(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_ROC(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_RSI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_VPT(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_VR(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_VRSI(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    indicator=QA.QA_indicator_VSTD(data)
    dataset=pd.concat([dataset,indicator],axis=1)

    #indicator=QA.QA_indicator_WR(data)
    #dataset=pd.concat([dataset,indicator],axis=1)
    
    return dataset

# def get_all_indicator(data):
#     dataset=pd.DataFrame()

#     indicator=QA.QA_indicator_RSI(data)
#     dataset=pd.concat([dataset,indicator],axis=1)
    
#     return dataset

def split_bin(data,bins=5):
    bins=pd.cut(data.iloc[:,0],bins,retbins=False)
    ont_hot=pd.get_dummies(bins)
    return ont_hot


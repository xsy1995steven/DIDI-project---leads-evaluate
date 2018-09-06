#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from imp import reload
reload(sys)
import os
import pandas as pd
import numpy as np
import operator 
import matplotlib.pyplot as plt
import time
import sklearn
from datetime import datetime
from datetime import timedelta
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn import ensemble, cross_validation  
from sklearn.model_selection import cross_val_predict 
from matplotlib import pyplot as pl
import sklearn.tree as tree
import pydotplus
import datetime
from IPython.display import Image
import matplotlib
import sklearn.tree as tree
# import pydotplus
# os.chdir("/Users/didi/Desktop/xsc/后市场/租售业务") 正样本占比2.5%

#读取文件，筛选人群
# df_origin = pd.read_csv('lyp_zs_users_info_features_all_add_info.csv',header=0,sep='\t')
# df_origin=df_origin[(df_origin['isthere_id']==1) | (df_origin['cars_3']>14)]
df = pd.read_csv('zs_users_plus_channel_id_feature.csv',sep = '\t',header=2,error_bad_lines=False)
print(df.columns)
colnames = df.columns
colnames_modified = []
for col in colnames:
    col = col.replace('x.','')
    col = col.replace('y.','')
    colnames_modified.append(col)
df.columns = colnames_modified
print(df.columns)

channel_ids = ['system','user_submit','reserve_submit','mis_import']
# df_origin = df[  (df['channel_id']=='system')
df_origin = df[ df['channel_id'].isin(channel_ids) ]
df_orogin = df_origin[df_origin['last_days'] <= 15]
data = df


# print(df_origin.columns)
feature=['dates_1', 'dates_2','dates_3','dates_4','last_days','cars_1', 'types_1', 'avg_amounts_1', 'min_amounts_1',
       'max_amounts_1',  'cars_2', 'types_2', 'avg_amounts_2',
       'min_amounts_2', 'max_amounts_2',  'cars_3', 'types_3',
       'brand_choice_3', 'engine_choice_3', 'plan_choice_3', 'avg_amounts_3',
       'min_amounts_3', 'max_amounts_3',  'cars_4', 'types_4',
       'brand_choice_4', 'engine_choice_4', 'plan_choice_4','isthere_name', 'isthere_id','sum_distinct_channel','sum_channel']
features=['dates_1', 'dates_2','dates_3','dates_4','last_days', 'cars_1', 'types_1', 'avg_amounts_1', 'min_amounts_1',
       'max_amounts_1',  'cars_2', 'types_2', 'avg_amounts_2',
       'min_amounts_2', 'max_amounts_2',  'cars_3', 'types_3',
       'brand_choice_3', 'engine_choice_3', 'plan_choice_3', 'avg_amounts_3',
       'min_amounts_3', 'max_amounts_3',  'cars_4', 'types_4',
       'brand_choice_4', 'engine_choice_4', 'plan_choice_4','isthere_name', 'isthere_id','sum_distinct_channel','sum_channel','is_yuyue']
data=data[features]
tmp=df_origin[df_origin['is_yuyue']==1]
tt=tmp[df_origin['yuyue_city_name']==df_origin['work_city']]

yuyue_df=df_origin[df_origin['is_yuyue']==1]
print('*****info about test data******')
print('same city ratio',len(tt)/(len(tmp)+0.001))
print('yuyue',len(df_origin[df_origin['is_yuyue']==1]))
print('not_yuyue',len(df_origin[df_origin['is_yuyue']==0]))
print('isthere_id',len(df_origin[(df_origin['isthere_id']==1)]),len(df_origin[(df_origin['isthere_id']==1) & (df_origin['is_yuyue']==0)]))
print('cars_3',len(df_origin[(df_origin['cars_3']>14)]),len(df_origin[(df_origin['cars_3']>14) & (df_origin['is_yuyue']==0)]))
print('yuyue_isthere_id',len(yuyue_df[yuyue_df['isthere_id']==1]),'yuyue_car_3',len(yuyue_df[yuyue_df['cars_3']>14]))
print('yuyue_union',len(yuyue_df[(yuyue_df['isthere_id']==1) | (yuyue_df['cars_3']>14)]))
print('\n')
print('\n')


#该评价指标用来评价模型好坏  
def rmspe(zip_list,count):  
    # w = ToWeight(y)  
    # rmspe = np.sqrt(np.mean((y - yhat) ** 2))  
    sum_value=0.0  
    # count=len(zip_list)  
    for real,predict in zip_list:  
        v1=(real-predict)**2  
        sum_value += v1  
    v2=sum_value / count  
    v3=np.sqrt(v2)  
    return v3  

def down_sample(df):
    df1=df[df['is_yuyue']==1]#正例
    df2=df[df['is_yuyue']==0]##负例
    df3=df2.sample(frac=0.2)##抽负例 使得保持1:1
    return pd.concat([df1,df3],ignore_index=True)
def up_sample(df):
    df1=df[df['is_yuyue']==1]#正例
    df2=df[df['is_yuyue']==0]##负例
    df3=df1
    df4=df2.sample(frac=0.99)
    # for x in xrange(0,100):
    # df3=pd.concat([df1,df1,df1,df1,df1,df1],ignore_index=True)
    return pd.concat([df4,df3],ignore_index=True)


# """
# #提取特征和目标值  
# data1_other,data1=cross_validation.train_test_split(data[data['is_yuyue']==1], test_size=0.2, random_state=10)
# data2=cross_validation.train_test_split(data[data['is_yuyue']==0],test_size=len(data1)/len(data[data['is_yuyue']==0]),random_state=10)
# data=data1.append(data2)
# data_xgb=down_sample(data)
# train_and_valid, test = cross_validation.train_test_split(data_xgb, test_size=0.3, random_state=10)  
# train, valid = cross_validation.train_test_split(train_and_valid, test_size=0.1, random_state=10)  
# train_feature, train_target = train[feature],train['is_yuyue']
# test_feature, test_target = test[feature],test['is_yuyue'] 

#data_other,data=cross_validation.train_test_split(data,test_size=0.001,random_state=10)#为了减少代码运行时间，方便测试  

# valid_feature, valid_target = valid[feature],valid['is_yuyue']
# params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 100,  
#           'learning_rate': 0.01, 'loss': 'ls'}  
# clf = ensemble.GradientBoostingRegressor(**params)  
# clf.fit(train_feature, train_target) #训练   
# pre=clf.predict(test_feature)  
# pre_list=list(pre)  
# real_pre_zip=zip(test_target,pre_list)  
# count=len(pre_list)  
# error=rmspe(real_pre_zip,count)  
# print(error) 
# y_pred = (pre >= 0.2)*1
# from sklearn import metrics
# print ('AUC: %.4f' % metrics.roc_auc_score(test['is_yuyue'],pre_list))
# print ('ACC: %.4f' % metrics.accuracy_score(test['is_yuyue'],y_pred))
# print ('Recall: %.4f' % metrics.recall_score(test['is_yuyue'],y_pred))
# print ('F1-score: %.4f' %metrics.f1_score(test['is_yuyue'],y_pred))
# print ('Precesion: %.4f' %metrics.precision_score(test['is_yuyue'],y_pred))
# print (metrics.confusion_matrix(test['is_yuyue'],y_pred))
# for x in range(0,len(feature)):  
#     print (feature[x],clf.feature_importances_[x])
# pre_all=clf.predict(df_origin[feature])  
# df_origin['pre']=pre_all
# df_origin.to_csv('data_all_pre_results.csv')


import xgboost as xgb

data_xgb=down_sample(data)


train_and_valid, test = cross_validation.train_test_split(data_xgb, test_size=0.3, random_state=10)  
train, valid = cross_validation.train_test_split(train_and_valid, test_size=0.1, random_state=10)  
train_feature, train_target = train[feature],train['is_yuyue']
test_feature, test_target = test[feature],test['is_yuyue'] 
dtrain=xgb.DMatrix(train_feature,label=train_target)
dtest=xgb.DMatrix(test[feature])
params={
'booster':'gbtree',
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth':4,
'lambda':10,
'subsample':0.75,
'colsample_bytree':0.75,
'min_child_weight':2, #min_child_weight [default=1]
'eta': 0.025,# eta [default=0.3, alias: learning_rate]
'seed':0,
'nthread':-1,
'silent':1
# 'gamma':0, #gamma [default=0, alias: min_split_loss
# 'scale_pos_weight':3,
# 'n_estimators':10
}
# params={
# 'booster':'gbtree',
# 'objective': 'binary:logistic',
# 'eval_metric': 'auc',
# 'max_depth':6,
# 'lambda':0.8,
# 'subsample':0.8,
# 'colsample_bytree':0.9,
# 'min_child_weight':1,
# 'eta': 0.01,
# 'seed':0,
# 'nthread':-1,
# 'n_estimators':20,
# 'silent':1,
# 'scale_pos_weight':6
# }

watchlist = [(dtrain,'train')]
bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist,early_stopping_rounds=30)

# #模型保存与显示
# bst.save_model('0001.model')
# xgb.to_graphviz(bst, num_trees=10)
# # dump model
bst.dump_model('dump.raw.txt')
# # dump model with feature map
# bst.dump_model('dump.nice.txt', 'featmap.txt')
# bst.save_model('xgb.model')
# bst2 = xgb.Booster(model_file='xgb.model')
# ypred=bst2.predict(dtest)
xgb.plot_tree(bst)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree2.png')

# 设置阈值, 输出一些评价指标
ypred=bst.predict(dtest)
y_pred = (ypred >= 0.5)*1
# print('y_pred',len(y_pred))
# print(y_pred.drop_duplicates(['']))
# 上采样之后的数据集上面进行结果测试
from sklearn import metrics
print('*********performance of model on the whole data after sampling********')
print ('AUC: %.4f' % metrics.roc_auc_score(test['is_yuyue'],ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test['is_yuyue'],y_pred))
print ('Recall: %.4f' % metrics.recall_score(test['is_yuyue'],y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test['is_yuyue'],y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test['is_yuyue'],y_pred))
print('\n')
print('\n')

metrics.confusion_matrix(test['is_yuyue'],y_pred)


# 原始数据上进行模型结果测试
# pre_all=clf.predict(df_origin[feature])  
# df_origin['pre']=pre_all
# df_origin.to_csv('data_all_pre_results_gbdt.csv')
preall=bst.predict(xgb.DMatrix(df_origin[feature]))
pre_all = (preall >= 0.5)*1
print("*********performance of data on test data from different channel**********")
print('pre_all',len(pre_all))
print ('AUC: %.4f' % metrics.roc_auc_score(df_origin['is_yuyue'],pre_all))
print ('ACC: %.4f' % metrics.accuracy_score(df_origin['is_yuyue'],pre_all))
print ('Recall: %.4f' % metrics.recall_score(df_origin['is_yuyue'],pre_all))
print ('F1-score: %.4f' %metrics.f1_score(df_origin['is_yuyue'],pre_all))
print ('Precesion: %.4f' %metrics.precision_score(df_origin['is_yuyue'],pre_all))
metrics.confusion_matrix(df_origin['is_yuyue'],pre_all)
# fig, ax = plt.subplots(figsize=(12,18))
# xgb.plot_importance(bst, max_num_features=50, height=0.8, ax=ax)
# plt.show()

df_origin['pre']=preall
# df_origin=df_origin.sort_values(by='pre',ascending=False)
# df_origin.to_csv('data_all_pre_results_xgboost.csv',sep=',',encoding='utf_8_sig')
res=df_origin[df_origin['is_yuyue']==0]
# res=res[['tel','pre','city_name','is_driver','pid','fast_driver_cluster_group','driver_online_time_30d','driver_order_ct_30d','driver_order_days_30d','driver_participate_way','isthere_id','cars_3']]
res = res[['tel','last_days','cars_3','sum_channel','sum_distinct_channel','isthere_name','max_amounts_3','avg_amounts_1','pre']]
res.groupby(['tel']).mean()#mean的取法似乎不太好
res= res[res['pre']>0.5]

# temp1=res[(res['isthere_id']==1) & (res['pre']>0.7)]
# temp2=res[(res['isthere_id']<1) & (res['pre']>0.5)]
# temp3=res[(res['isthere_id']<1) & (res['cars_3']<15)]
# temp3=res[res['cars_3']>15]
print('res',len(res))
# print('temp1',len(temp1),len(temp1)/(len(res)+0.01),temp1['pre'].mean())
# print('temp2',len(temp2),len(temp2)/(len(res)+0.01),temp2['pre'].mean())
res=res.sort_values(by='pre',ascending=False)

res.to_excel('result_channel_file/data_part_pre_results_xgboost0716_system.xlsx',  index=False,encoding='utf-8',sheet_name='Sheet')

res.to_csv('result_channel_file/data_part_pre_results_xgboost0716_system.csv',sep=',',encoding='utf_8_sig')
# df_origin.to_csv('data_all_pre_results_xgboost.csv')
# """'''
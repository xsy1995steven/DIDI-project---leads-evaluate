import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib

#read provided data,get df
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
df = df[  (df['channel_id']=='system') ]

#define features
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

data = df[features]

#define sample functions
def up_sample(df, fac):
    df1 = df[df['is_yuyue']==1]
    df2 = df[df['is_yuyue']==0]
    df1 = pd.concat([df1 for x in range(int(fac))], ignore_index=True)
    return pd.concat([df1,df2],ignore_index=True)

def down_sample(df, fac):
    df1 = df[df['is_yuyue']==1]
    df2 = df[df['is_yuyue']==0]
    df2 = df.sample(frac=fac)
    return pd.concat([df1,df2],ignore_index=True)

import xgboost as xgb

#get trainning data
xgb_data = down_sample(data, 0.25)
train_valid, test = train_test_split(xgb_data, test_size=0.2, random_state=10)
train, valid = train_test_split(train_valid, test_size=0.2, random_state=10)
train_feature, train_target = train[feature], train['is_yuyue']
test_feature, test_target = test[feature], test['is_yuyue']
dtrain = xgb.DMatrix(train_feature, label=train_target)
dtest = xgb.DMatrix(test_feature)
params = {
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
watchlist = [(dtrain,'train')]
bst = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist,early_stopping_rounds=30)

#模型保存与显示
bst.save_model('xgb_model.model')
bst.dump_model('xgb_model.raw.txt')
xgb.plot_tree(bst)
fig = matplotlib.pyplot.gcf()#Get a reference to the current figure
fig.set_size_inches(150, 100)
fig.savefig('tree2.png')

#在测试集上测试
from sklearn import metrics
y_pred = bst.predict(dtest)
ypred = (y_pred >= 0.5)*1
print('AUC:%s\n'%metrics.roc_auc_score(test_target,y_pred))
print('auccuracy:%s\n'%metrics.accuracy_score(test_target,ypred))
print('recall:%s\n'%metrics.recall_score(test_target,ypred))
print('precision:%s\n'%metrics.precision_score(test_target,ypred))
print('F1-score:%s\n'%metrics.f1_score(test_target,ypred))

#在原始数据上测试
dtest_whole = xgb.DMatrix(df[feature])
pre_all = bst.predict(dtest_whole)
preall = (pre_all>=0.5)*1
test_target = df['is_yuyue']
print('AUC:%s\n'%metrics.roc_auc_score(test_target,pre_all))
print('auccuracy:%s\n'%metrics.accuracy_score(test_target,preall))
print('recall:%s\n'%metrics.recall_score(test_target,preall))
print('precision:%s\n'%metrics.precision_score(test_target,preall))
print('F1-score:%s\n'%metrics.f1_score(test_target,preall))

#排序并输出top100
df['pre'] = pre_all
res = df[df['is_yuyue'] == 0]
res = res[res['pre']>0.5]
res = res.groupby(['pid']).mean()
res = res.sort_values(by='pre',ascending=False)

res.to_excel('result_channel_file/result_by_self.xlsx',  index=False,encoding='utf-8',sheet_name='Sheet')

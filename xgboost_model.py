"""

This is the gradient boosting model using XGBoost for "TalkingData AdTracking Fraud Detection Challenge".
The model is revised from "https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680".

Features used:
    ip:               ip address of click.
    app:              app id for marketing.
    device:           device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
    os:               os version id of user mobile phone
    channel:          channel id of mobile ad publisher
    click_time:       timestamp of click (UTC)
    day:              day of click, derived from click_time
    hour:             hour of click, derived from click_time
    dayofweek:        day of week of click, derived from click_time
    clicks_by_ip:     number of clicks per IP
    
Target:
    is_attributed:    target that is to be predicted, indicating the app was downloaded
    
The evaluation is based on AUC-ROC score.
This model gets score 0.9648 in test test.
	
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import gc
import time
import datetime

# Debug flag
debug = False

# Parameters for XGBoost model
params = {'eta': 0.6,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}

# Features used
train_cols = ['ip','app','device','os','channel','click_time','is_attributed']
test_cols = ['ip','app','device','os','channel','click_time','click_id']
dtypes = {
	'ip'		:'uint32',
	'app'		:'uint16',
	'device'	:'uint16',
	'os'		:'uint16',
	'channel'	:'uint16',
	'is_attributed'	:'uint8',
	'click_id'	:'uint32'
	}
if not debug:
	train = pd.read_csv("train.csv",usecols=train_cols,dtype=dtypes,skiprows=range(1,124903891),nrows=60000000)
	test = pd.read_csv("test.csv",usecols=test_cols,dtype=dtypes)
# use part of training set to ensure validity of model
else:
	train = pd.read_csv("train.csv",nrows=1000000,usecols=train_cols,dtype=dtypes)
	test = pd.read_csv("test.csv",nrows=100000,usecols=test_cols,dtype=dtypes)

y = train['is_attributed']
train.drop(['is_attributed'],axis=1,inplace=True)

# Output dataframe
out = pd.DataFrame()
out['click_id'] = test['click_id'].astype('int')

test.drop(['click_id'],axis=1,inplace=True)
num_train = train.shape[0]
merge = pd.concat([train,test])

del train,test
gc.collect()

ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip','clicks_by_ip']
merge = merge.merge(ip_count,on='ip',how='left',sort=False)
merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
merge['datetime'] = pd.to_datetime(merge['click_time'])
merge['dayofweek'] = merge['datetime'].dt.dayofweek
merge['dayofyear'] = merge['datetime'].dt.dayofyear
merge['hour'] = merge['datetime'].dt.hour
merge.drop(['ip','click_time','datetime'],axis=1,inplace=True)

train = merge[:num_train]
test = merge[num_train:]

dtrain = xgb.DMatrix(train,y)
del train,y
gc.collect()
watchlist = [(dtrain,'train')]
# Construct model
model = xgb.train(params,dtrain,15,watchlist,maximize=True,verbose_eval=1)

del dtrain
gc.collect()

dtest = xgb.DMatrix(test)
del test
gc.collect()

# Predict and output
out['is_attributed'] = model.predict(dtest,ntree_limit=model.best_ntree_limit)
out.to_csv('submission.csv',index=False)

"""

This is the deep learning model for "TalkingData AdTracking Fraud Detection Challenge".
The model is revised from "https://www.kaggle.com/alexanderkireev/experiments-with-imbalance-nn-arch-9728".
At runtime, this takes 7.54G training data and about 22G RAM.

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
    ip_day_hour:      number of clicks per IP,day,hour
    ip_app_count:     number of clicks per IP,app
    ip_app_os_count:  number of clicks per IP,app,os
    
Target:
    is_attributed:    target that is to be predicted, indicating the app was downloaded
    
The evaluation is based on AUC-ROC score.
This model gets score 0.9669 in test test.
    

"""

import os
import gc         # since it is memory-intense, we need gc to release memory as long as we don't need them
import time
import datetime
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv1D, Embedding, Input, concatenate


# Debug flag
debug = False
batch_size = 20000      # batch size for the model

# features used
train_cols = ['ip','app','device','os','channel','click_time','is_attributed']
test_cols = ['ip','app','device','os','channel','click_time','click_id']
dtypes = {
    'ip'        :'uint32',
    'app'        :'uint16',
    'device'    :'uint16',
    'os'        :'uint16',
    'channel'    :'uint16',
    'is_attributed'    :'uint8',
    'click_id'    :'uint32'
    }

# We only need last 60 million data for training
train = pd.read_csv("train.csv",
                    usecols=train_cols,
                    dtype=dtypes,
                    skiprows=range(1,124903891),
                    nrows=60000000)
test = pd.read_csv("test.csv",
                   usecols=test_cols,
                   dtype=dtypes)

print("Preparing Data...")
y = train['is_attributed'].values
train.drop(['is_attributed'],axis=1,inplace=True)

# Output dataframe
out = pd.DataFrame()
out['click_id'] = test['click_id'].astype('int')

test.drop(['click_id'],axis=1,inplace=True)
num_train = train.shape[0]
merge = pd.concat([train,test])

del train,test
gc.collect()

# Time features: day of week, day and hour
print("Generating time features...")
merge['datetime'] = pd.to_datetime(merge['click_time'])
merge['dayofweek'] = merge['datetime'].dt.dayofweek.astype('uint8')
merge['day'] = merge['datetime'].dt.day.astype('uint8')
merge['hour'] = merge['datetime'].dt.hour.astype('uint8')

print("Generating clicks by ip feature...")
ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip','clicks_by_ip']
merge = merge.merge(ip_count,on='ip',how='left')
merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
del ip_count
gc.collect()

print("Generating ip-day-hour feature...")
ip_day_hour = merge[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour'})
merge = merge.merge(ip_day_hour, on=['ip','day','hour'], how='left')
del ip_day_hour
gc.collect()

print("Generating ip-app feature...")
ip_app = merge[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
merge = merge.merge(ip_app, on=['ip','app'], how='left')
del ip_app
gc.collect()

print("Generating ip-app-os feature...")
ip_app_os = merge[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
merge = merge.merge(ip_app_os, on=['ip','app', 'os'], how='left')
del ip_app_os
gc.collect()

merge['ip_day_hour'] = merge['ip_day_hour'].astype('uint16')
merge['ip_app_count'] = merge['ip_app_count'].astype('uint16')
merge['ip_app_os_count'] = merge['ip_app_os_count'].astype('uint16')
merge.drop(['ip','click_time','datetime'],axis=1,inplace=True)

# for constructing embedding layer
max_app = merge['app'].max()+1
max_dev = merge['device'].max()+1
max_os = merge['os'].max()+1
max_ch = merge['channel'].max()+1
max_dow = merge['dayofweek'].max()+1
max_day = merge['day'].max()+1
max_hr = merge['hour'].max()+1
max_clk = merge['clicks_by_ip'].max()+1
max_idh = merge['ip_day_hour'].max()+1
max_iac = merge['ip_app_count'].max()+1
max_iaoc = merge['ip_app_os_count'].max()+1

train = merge[:num_train]
test = merge[num_train:]

X = {
    'app': np.array(train.app),
    'ch': np.array(train.channel),
    'dev': np.array(train.device),
    'os': np.array(train.os),
    'hr': np.array(train.hour),
    'day': np.array(train.day),
    'dow': np.array(train.dayofweek),
    'clk': np.array(train.clicks_by_ip),
    'idh': np.array(train.ip_day_hour),
    'iac': np.array(train.ip_app_count),
    'iaoc': np.array(train.ip_app_os_count)
    }

del train
gc.collect()

# Now it's time to train
print("Constructing model...")
n_embed = 50

# Construct the model
in_app = Input(shape=[1], name = 'app')
in_dev = Input(shape=[1], name = 'dev')
in_os = Input(shape=[1], name = 'os')
in_ch = Input(shape=[1], name = 'ch')
in_dow = Input(shape=[1], name = 'dow')
in_day = Input(shape=[1], name = 'day')
in_hr = Input(shape=[1], name = 'hr')
in_clk = Input(shape=[1], name = 'clk')
in_idh = Input(shape=[1], name = 'idh')
in_iac = Input(shape=[1], name = 'iac')
in_iaoc = Input(shape=[1], name = 'iaoc')
embed_app = Embedding(max_app,n_embed)(in_app)
embed_dev = Embedding(max_dev,n_embed)(in_dev)
embed_os = Embedding(max_os,n_embed)(in_os)
embed_ch = Embedding(max_ch,n_embed)(in_ch)
embed_dow = Embedding(max_dow,n_embed)(in_dow)
embed_day = Embedding(max_day,n_embed)(in_day)
embed_hr = Embedding(max_hr,n_embed)(in_hr)
embed_clk = Embedding(max_clk,n_embed)(in_clk)
embed_idh = Embedding(max_idh,n_embed)(in_idh)
embed_iac = Embedding(max_iac,n_embed)(in_iac)
embed_iaoc = Embedding(max_iaoc,n_embed)(in_iaoc)
concat = concatenate([(embed_app),(embed_dev),(embed_os),(embed_ch),(embed_dow),(embed_day),(embed_hr),(embed_clk)
                      ,(embed_idh),(embed_iac),(embed_iaoc)])
fl = Flatten()(concat)

# We use 5-layer neural network here
d1 = Dense(512,activation='relu')(fl)
dp1 = Dropout(0.2)(d1)
d2 = Dense(512,activation='relu')(dp1)
dp2 = Dropout(0.2)(d2)
d3 = Dense(128,activation='relu')(dp2)
dp3 = Dropout(0.2)(d3)
d4 = Dense(128,activation='relu')(dp3)
dp4 = Dropout(0.2)(d4)
d5 = Dense(32,activation='relu')(dp4)
dp5 = Dropout(0.2)(d5)
output = Dense(1,activation='sigmoid')(dp5)
model = Model(inputs=[in_app,in_dev,in_os,in_ch,in_dow,in_day,in_hr,in_clk,in_idh,in_iac,in_iaoc],outputs=output)


# Compile the model
print("Compiling the model...")
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# Prediction
if debug is True:
    print("Start fitting...")
    model.fit(X,y,
              batch_size=batch_size, 
              epochs=1,                         
              verbose=1,
              validation_split = 0.2,
              class_weight={0:0.01,1:0.99})     # Since label class is imbalanced, it is set to compensate this
else:
    print("Start fitting...")
    model.fit(X,y,
              batch_size=batch_size, 
              epochs=1,                         # It seems more epochs doesn't ring us much effect, so just set it 1
              verbose=1,
              class_weight={0:0.01,1:0.99})     # Since label class is imbalanced, it is set to compensate this
    
    X_test = {
        'app': np.array(test.app),
        'ch': np.array(test.channel),
        'dev': np.array(test.device),
        'os': np.array(test.os),
        'hr': np.array(test.hour),
        'day': np.array(test.day),
        'dow': np.array(test.dayofweek),
        'clk': np.array(test.clicks_by_ip),
        'idh': np.array(test.ip_day_hour),
        'iac': np.array(test.ip_app_count),
        'iaoc': np.array(test.ip_app_os_count)
    }
    del test
    gc.collect()
    
    print("Predicting...")
    out['is_attributed'] = model.predict(X_test,batch_size=batch_size, verbose=1)
    out.to_csv('submission.csv',index=False)
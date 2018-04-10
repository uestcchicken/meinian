import pandas as pd 
import numpy as np 
import lightgbm as lgb 
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import xgboost as xgb
import math

def my_error(preds, train_data):
    labels = train_data.get_label()
    result = np.mean((np.log10(preds + 1) - np.log10(labels + 1)) ** 2)
    return 'my_error', result, False

train_path = 'train_cutlose.csv'
test_path = 'test_cutlose.csv'

print('loading train data...')
train = pd.read_csv(train_path)
print('loading test data...')
test = pd.read_csv(test_path)

print(train.info())
print(train.sample(20))

l = len(train)
r = 0.1
train = train.sample(frac = 1)
val = train[(l - round(r * l)):]
train = train[:(l - round(r * l))]

predictors = list(train.columns)
predictors.remove('vid')
for t in ['A', 'B', 'C', 'D', 'E']:
    predictors.remove(t)
print(predictors)

params = {
    'boosting': 'gbdt',
    'metric': 'rmse',
    'application': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'nthread': 8
}

dtrain_A = lgb.Dataset(train[predictors].values, label = train['A'].values, feature_name = predictors)
dvalid_A = lgb.Dataset(val[predictors].values, label = val['A'].values, feature_name = predictors)
dtrain_B = lgb.Dataset(train[predictors].values, label = train['B'].values, feature_name = predictors)
dvalid_B = lgb.Dataset(val[predictors].values, label = val['B'].values, feature_name = predictors)
dtrain_C = lgb.Dataset(train[predictors].values, label = train['C'].values, feature_name = predictors)
dvalid_C = lgb.Dataset(val[predictors].values, label = val['C'].values, feature_name = predictors)
dtrain_D = lgb.Dataset(train[predictors].values, label = train['D'].values, feature_name = predictors)
dvalid_D = lgb.Dataset(val[predictors].values, label = val['D'].values, feature_name = predictors)
dtrain_E = lgb.Dataset(train[predictors].values, label = train['E'].values, feature_name = predictors)
dvalid_E = lgb.Dataset(val[predictors].values, label = val['E'].values, feature_name = predictors)
lgb_model_A = lgb.train(params, dtrain_A, verbose_eval = 20, valid_sets = [dtrain_A, dvalid_A], num_boost_round = 400, early_stopping_rounds = 30)
lgb_model_B = lgb.train(params, dtrain_B, verbose_eval = 20, valid_sets = [dtrain_B, dvalid_B], num_boost_round = 400, early_stopping_rounds = 30)
lgb_model_C = lgb.train(params, dtrain_C, verbose_eval = 20, valid_sets = [dtrain_C, dvalid_C], num_boost_round = 400, early_stopping_rounds = 30)
lgb_model_D = lgb.train(params, dtrain_D, verbose_eval = 20, valid_sets = [dtrain_D, dvalid_D], num_boost_round = 400, early_stopping_rounds = 30)
lgb_model_E = lgb.train(params, dtrain_E, verbose_eval = 20, valid_sets = [dtrain_E, dvalid_E], num_boost_round = 400, early_stopping_rounds = 30)

print('predicting submission...')
submit = pd.read_csv(test_path, usecols = ['vid'])       
submit['A'] = lgb_model_A.predict(test[predictors], num_iteration = lgb_model_A.best_iteration)
submit['B'] = lgb_model_B.predict(test[predictors], num_iteration = lgb_model_B.best_iteration)
submit['C'] = lgb_model_C.predict(test[predictors], num_iteration = lgb_model_C.best_iteration)
submit['D'] = lgb_model_D.predict(test[predictors], num_iteration = lgb_model_D.best_iteration)
submit['E'] = lgb_model_E.predict(test[predictors], num_iteration = lgb_model_E.best_iteration)

'''
print('AFeature names:', lgb_model_A.feature_name())
print('AFeature importances:', list(lgb_model_A.feature_importance()))      
print('BFeature names:', lgb_model_B.feature_name())
print('BFeature importances:', list(lgb_model_B.feature_importance()))  
print('CFeature names:', lgb_model_C.feature_name())
print('CFeature importances:', list(lgb_model_C.feature_importance()))  
print('DFeature names:', lgb_model_D.feature_name())
print('DFeature importances:', list(lgb_model_D.feature_importance()))  
print('EFeature names:', lgb_model_E.feature_name())
print('EFeature importances:', list(lgb_model_E.feature_importance()))  
'''

print('writing submission...')
submit.to_csv('submission.csv', index = False, header = False)
print('done.')


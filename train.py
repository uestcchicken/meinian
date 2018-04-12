import pandas as pd 
import numpy as np 
import lightgbm as lgb 
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import xgboost as xgb
import math

def obj_function(preds, train_data):
    y = train_data.get_label()
    p = preds
    grad = 2.0 / (p + 1.0) * (np.log1p(p) - np.log1p(y))
    hess = 2.0 / np.square(p + 1.0) * (1.0 - np.log1p(p) + np.log1p(y))
    return grad, hess
    
def eval_function(preds, train_data):
    labels = train_data.get_label()
    return 'loss', np.mean(np.square(np.log1p(preds) - np.log1p(labels))), False

train_path = 'train_cut_useless.csv'
test_path = 'test_cutlose.csv'

print('loading train data...')
train = pd.read_csv(train_path)
print('loading test data...')
test = pd.read_csv(test_path)

print(train.info())
print(train.sample(5))

l = len(train)
r = 0.1
val = train[(l - round(r * l)):]
train = train[:(l - round(r * l))]

predictors = list(train.columns)
predictors.remove('vid')
for t in ['A', 'B', 'C', 'D', 'E']:
    predictors.remove(t)
#print(predictors)

categorical = ['2302', '0116', '0113_1', '0113_2']

params = {
    'boosting': 'gbdt',
    'objective': 'none',
    #'application': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'nthread': 8
}

dtrain_A = lgb.Dataset(train[predictors].values, label = train['A'].values, feature_name = predictors, categorical_feature = categorical)
dvalid_A = lgb.Dataset(val[predictors].values, label = val['A'].values, feature_name = predictors, categorical_feature = categorical)
dtrain_B = lgb.Dataset(train[predictors].values, label = train['B'].values, feature_name = predictors, categorical_feature = categorical)
dvalid_B = lgb.Dataset(val[predictors].values, label = val['B'].values, feature_name = predictors, categorical_feature = categorical)
dtrain_C = lgb.Dataset(train[predictors].values, label = train['C'].values, feature_name = predictors, categorical_feature = categorical)
dvalid_C = lgb.Dataset(val[predictors].values, label = val['C'].values, feature_name = predictors, categorical_feature = categorical)
dtrain_D = lgb.Dataset(train[predictors].values, label = train['D'].values, feature_name = predictors, categorical_feature = categorical)
dvalid_D = lgb.Dataset(val[predictors].values, label = val['D'].values, feature_name = predictors, categorical_feature = categorical)
dtrain_E = lgb.Dataset(train[predictors].values, label = train['E'].values, feature_name = predictors, categorical_feature = categorical)
dvalid_E = lgb.Dataset(val[predictors].values, label = val['E'].values, feature_name = predictors, categorical_feature = categorical)


result_A = {}
result_B = {}
result_C = {}
result_D = {}
result_E = {}

lgb_model_A = lgb.train(params, dtrain_A, feval = eval_function, fobj = obj_function, evals_result = result_A, verbose_eval = 50, \
    valid_sets = [dtrain_A, dvalid_A], num_boost_round = 1000, early_stopping_rounds = 50)
lgb_model_B = lgb.train(params, dtrain_B, feval = eval_function, fobj = obj_function, evals_result = result_B, verbose_eval = 50, \
    valid_sets = [dtrain_B, dvalid_B], num_boost_round = 1000, early_stopping_rounds = 50)
lgb_model_C = lgb.train(params, dtrain_C, feval = eval_function, fobj = obj_function, evals_result = result_C, verbose_eval = 50, \
    valid_sets = [dtrain_C, dvalid_C], num_boost_round = 1000, early_stopping_rounds = 50)
lgb_model_D = lgb.train(params, dtrain_D, feval = eval_function, fobj = obj_function, evals_result = result_D, verbose_eval = 50, \
    valid_sets = [dtrain_D, dvalid_D], num_boost_round = 1000, early_stopping_rounds = 50)
lgb_model_E = lgb.train(params, dtrain_E, feval = eval_function, fobj = obj_function, evals_result = result_E, verbose_eval = 50, \
    valid_sets = [dtrain_E, dvalid_E], num_boost_round = 1000, early_stopping_rounds = 50)

loss_all = 0.0
loss_all += result_A['valid_1']['loss'][lgb_model_A.best_iteration - 1]
loss_all += result_B['valid_1']['loss'][lgb_model_B.best_iteration - 1]
loss_all += result_C['valid_1']['loss'][lgb_model_C.best_iteration - 1]
loss_all += result_D['valid_1']['loss'][lgb_model_D.best_iteration - 1]
loss_all += result_E['valid_1']['loss'][lgb_model_E.best_iteration - 1]

print('predicting submission...')
submit = pd.read_csv(test_path, usecols = ['vid'])       
submit['A'] = lgb_model_A.predict(test[predictors], num_iteration = lgb_model_A.best_iteration)
submit['B'] = lgb_model_B.predict(test[predictors], num_iteration = lgb_model_B.best_iteration)
submit['C'] = lgb_model_C.predict(test[predictors], num_iteration = lgb_model_C.best_iteration)
submit['D'] = lgb_model_D.predict(test[predictors], num_iteration = lgb_model_D.best_iteration)
submit['E'] = lgb_model_E.predict(test[predictors], num_iteration = lgb_model_E.best_iteration)


print('Feature names:', lgb_model_A.feature_name())
print('AFeature importances:', list(lgb_model_A.feature_importance()))      
print('BFeature importances:', list(lgb_model_B.feature_importance()))  
print('CFeature importances:', list(lgb_model_C.feature_importance()))  
print('DFeature importances:', list(lgb_model_D.feature_importance()))  
print('EFeature importances:', list(lgb_model_E.feature_importance()))  

print('final loss: ', loss_all / 5)

print('writing submission...')
submit.to_csv('submission.csv', index = False, header = False)
print('done.')


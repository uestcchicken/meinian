import pandas as pd 
import numpy as np 
import lightgbm as lgb 
import gc
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

categorical = ['2302', '0116', '0113_1', '0113_2', '1001', '0118_1', '0118_2', '0437', \
    '0434_1', '0434_2', '0434_3', '0434_4', '0434_5', '0434_6', \
    '1402_1', '1402_2', '1402_3', '4001_1', '4001_2', \
    '0409_1', '0409_2', '0409_3', '0409_4', '0409_5', \
    '0409_6', '0409_7', '0409_8', '0409_9', '0409_10', \
    '3301', '3399', '30007']

params_num_leaves = [31, 63, 127]
params_min_data_in_leaf = [20, 60, 100]
params_max_bin = [127, 255, 511]
params_feature_fraction = [0.8]

params_num_leaves = [255]
params_min_data_in_leaf = [20]
params_max_bin = [127, 255, 511]
params_feature_fraction = [0.8]

best_loss_A = 100
best_loss_B = 100
best_loss_C = 100
best_loss_D = 100
best_loss_E = 100

best_loss_A = 0.01380204339311637
best_num_leaves_A = 63
best_min_data_in_leaf_A = 100
best_max_bin_A = 127
best_feature_fraction_A = 0.8
best_loss_B = 0.01767496008382642
best_num_leaves_B = 31
best_min_data_in_leaf_B = 100
best_max_bin_B = 255
best_feature_fraction_B = 0.8
best_loss_C = 0.07004568801289758
best_num_leaves_C = 63
best_min_data_in_leaf_C = 20
best_max_bin_C = 511
best_feature_fraction_C = 0.8
best_loss_D = 0.010433867588869343
best_num_leaves_D = 127
best_min_data_in_leaf_D = 60
best_max_bin_D = 255
best_feature_fraction_D = 0.8
best_loss_E = 0.029003656812253233
best_num_leaves_E = 63
best_min_data_in_leaf_E = 20
best_max_bin_E = 255
best_feature_fraction_E = 0.8


for num_leaves in params_num_leaves:
    for min_data_in_leaf in params_min_data_in_leaf:
        for max_bin in params_max_bin:
            for feature_fraction in params_feature_fraction:
                print('********************************************')
                print(num_leaves)
                print(min_data_in_leaf)
                print(max_bin)
                print(feature_fraction)
                params = {
                    'boosting': 'gbdt',
                    'objective': 'none',
                    'nthread': 8,
                    'learning_rate': 0.02,
                    'num_leaves': num_leaves,
                    'feature_fraction': feature_fraction,
                    'min_data_in_leaf': min_data_in_leaf,
                    'max_bin': max_bin,
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

                lgb_model_A = lgb.train(params, dtrain_A, feature_name = predictors, categorical_feature = categorical, \
                    feval = eval_function, fobj = obj_function, evals_result = result_A, verbose_eval = 200, \
                    valid_sets = [dtrain_A, dvalid_A], num_boost_round = 100000, early_stopping_rounds = 50)
                lgb_model_B = lgb.train(params, dtrain_B, feature_name = predictors, categorical_feature = categorical, \
                    feval = eval_function, fobj = obj_function, evals_result = result_B, verbose_eval = 200, \
                    valid_sets = [dtrain_B, dvalid_B], num_boost_round = 100000, early_stopping_rounds = 50)
                lgb_model_C = lgb.train(params, dtrain_C, feature_name = predictors, categorical_feature = categorical, \
                    feval = eval_function, fobj = obj_function, evals_result = result_C, verbose_eval = 200, \
                    valid_sets = [dtrain_C, dvalid_C], num_boost_round = 100000, early_stopping_rounds = 50)
                lgb_model_D = lgb.train(params, dtrain_D, feature_name = predictors, categorical_feature = categorical, \
                    feval = eval_function, fobj = obj_function, evals_result = result_D, verbose_eval = 200, \
                    valid_sets = [dtrain_D, dvalid_D], num_boost_round = 100000, early_stopping_rounds = 50)
                lgb_model_E = lgb.train(params, dtrain_E, feature_name = predictors, categorical_feature = categorical, \
                    feval = eval_function, fobj = obj_function, evals_result = result_E, verbose_eval = 200, \
                    valid_sets = [dtrain_E, dvalid_E], num_boost_round = 100000, early_stopping_rounds = 50)

                
                loss_A = result_A['valid_1']['loss'][lgb_model_A.best_iteration - 1]
                loss_B = result_B['valid_1']['loss'][lgb_model_B.best_iteration - 1]
                loss_C = result_C['valid_1']['loss'][lgb_model_C.best_iteration - 1]
                loss_D = result_D['valid_1']['loss'][lgb_model_D.best_iteration - 1]
                loss_E = result_E['valid_1']['loss'][lgb_model_E.best_iteration - 1]
                
                if loss_A < best_loss_A:
                    best_loss_A = loss_A
                    best_num_leaves_A = num_leaves
                    best_min_data_in_leaf_A = min_data_in_leaf
                    best_max_bin_A = max_bin
                    best_feature_fraction_A = feature_fraction
                if loss_B < best_loss_B:
                    best_loss_B = loss_B
                    best_num_leaves_B = num_leaves
                    best_min_data_in_leaf_B = min_data_in_leaf
                    best_max_bin_B = max_bin
                    best_feature_fraction_B = feature_fraction
                if loss_C < best_loss_C:
                    best_loss_C = loss_C
                    best_num_leaves_C = num_leaves
                    best_min_data_in_leaf_C = min_data_in_leaf
                    best_max_bin_C = max_bin
                    best_feature_fraction_C = feature_fraction
                if loss_D < best_loss_D:
                    best_loss_D = loss_D
                    best_num_leaves_D = num_leaves
                    best_min_data_in_leaf_D = min_data_in_leaf
                    best_max_bin_D = max_bin
                    best_feature_fraction_D = feature_fraction
                if loss_E < best_loss_E:
                    best_loss_E = loss_E
                    best_num_leaves_E = num_leaves
                    best_min_data_in_leaf_E = min_data_in_leaf
                    best_max_bin_E = max_bin
                    best_feature_fraction_E = feature_fraction
                
                print('best_loss_A =', best_loss_A)
                print('best_num_leaves_A =', best_num_leaves_A)
                print('best_min_data_in_leaf_A =', best_min_data_in_leaf_A)
                print('best_max_bin_A =', best_max_bin_A)            
                print('best_feature_fraction_A =', best_feature_fraction_A)
                
                print('best_loss_B =', best_loss_B)
                print('best_num_leaves_B =', best_num_leaves_B)
                print('best_min_data_in_leaf_B =', best_min_data_in_leaf_B)
                print('best_max_bin_B =', best_max_bin_B)            
                print('best_feature_fraction_B =', best_feature_fraction_B)
                
                print('best_loss_C =', best_loss_C)
                print('best_num_leaves_C =', best_num_leaves_C)
                print('best_min_data_in_leaf_C =', best_min_data_in_leaf_C)
                print('best_max_bin_C =', best_max_bin_C)            
                print('best_feature_fraction_C =', best_feature_fraction_C)

                print('best_loss_D =', best_loss_D)
                print('best_num_leaves_D =', best_num_leaves_D)
                print('best_min_data_in_leaf_D =', best_min_data_in_leaf_D)
                print('best_max_bin_D =', best_max_bin_D)            
                print('best_feature_fraction_D =', best_feature_fraction_D)

                print('best_loss_E =', best_loss_E)
                print('best_num_leaves_E =', best_num_leaves_E)
                print('best_min_data_in_leaf_E =', best_min_data_in_leaf_E)
                print('best_max_bin_E =', best_max_bin_E)            
                print('best_feature_fraction_E =', best_feature_fraction_E)
                
   
'''
best_loss_A: 0.01380204339311637
best_num_leaves_A: 63
best_min_data_in_leaf_A: 100
best_max_bin_A: 127
best_feature_fraction_A: 0.8
best_loss_B: 0.01767496008382642
best_num_leaves_B: 31
best_min_data_in_leaf_B: 100
best_max_bin_B: 255
best_feature_fraction_B: 0.8
best_loss_C: 0.07004568801289758
best_num_leaves_C: 63
best_min_data_in_leaf_C: 20
best_max_bin_C: 511
best_feature_fraction_C: 0.8
best_loss_D: 0.010433867588869343
best_num_leaves_D: 127
best_min_data_in_leaf_D: 60
best_max_bin_D: 255
best_feature_fraction_D: 0.8
best_loss_E: 0.029003656812253233
best_num_leaves_E: 63
best_min_data_in_leaf_E: 20
best_max_bin_E: 255
best_feature_fraction_E: 0.8
********************************************

'''
            
            

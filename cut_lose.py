import pandas as pd 
import numpy as np 
import gc

train_path = 'train_feature.csv'
test_path = 'test_feature.csv'

print('loading train data...')
train = pd.read_csv(train_path)
print('loading test data...')
test = pd.read_csv(test_path)

l = len(train)
train = train.append(test)
del test
gc.collect()

print(train.info())

drop_columns = []
for name in list(train.columns):
    n = train[name]
    values = n.isnull().value_counts()
    if len(values.index) == 1:
        if values.index[0] == True:
            drop_columns.append(name)
        continue
    if values[True] > 47500:
        drop_columns.append(name)
        
print(drop_columns)
train.drop(drop_columns, axis = 1, inplace = True)

print(train.info())

test = train[l:]
train = train[:l]

train.to_csv('train_cutlose.csv', index = False)
test.to_csv('test_cutlose.csv', index = False)
        

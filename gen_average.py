import pandas as pd 

train = pd.read_csv('train.csv', usecols = ['A', 'B', 'C', 'D', 'E'], dtype = 'float64')
test = pd.read_csv('test.csv')

print(train.mean())

a = 126.049358
b = 79.641870
c = 1.616626
d = 1.406719
e = 2.769653

test['A'] = a 
test['B'] = b 
test['C'] = c 
test['D'] = d 
test['E'] = e

test.to_csv('average.csv', index = False, header = False)

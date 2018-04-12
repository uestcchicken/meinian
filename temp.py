import pandas as pd 

f = pd.read_csv('train_cut_useless.csv')
f2 = pd.read_csv('test_cutlose.csv')

print(f['2406'].describe())
print(f2['2406'].describe())
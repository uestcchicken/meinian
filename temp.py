import pandas as pd 

f = pd.read_csv('train_cutlose.csv')

print(f['E'].describe())
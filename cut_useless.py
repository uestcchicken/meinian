import pandas as pd 

f = pd.read_csv('train_cutlose.csv')
print(f.info())


f = f[f['A'] > 0.0]
f = f[f['B'] > 0.0]
f = f[f['E'] > 0.0]
f = f[f['2403'] < 5000]
f = f[f['10004'] >= 0.0]
f = f[f['1814'] < 1800]

for t in list(f.columns):
    print(f[t].describe())
    
#f = f.sample(frac = 1)
    
f.to_csv('train_cut_useless.csv', index = False)
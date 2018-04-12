import pandas as pd 
import numpy as np

f = pd.read_csv('train_cut_useless.csv')


loss_all = 0
for name in ['A', 'B', 'C', 'D', 'E']:

    print(f[name].describe())
    mean = f[name].mean()

    nums = f[name].values

    sum = 0
    for n in nums:
        loss = np.square(np.log1p(mean) - np.log1p(n))
        sum += loss
    print(sum / len(nums))
    loss_all += sum / len(nums)
print(loss_all / 5)
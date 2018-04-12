import pandas as pd 

f = open('data.csv', 'r')
lines = f.readlines()[1:]
lines = [l[:-1].split('$') for l in lines]

train = pd.read_csv('train_feature.csv')
test = pd.read_csv('test_feature.csv')

len_train = len(train)
print(len(train))

train = train.append(test)
print(len(train))

for i in range(0, len(lines)):
    if i % 100000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]

    if l[1] not in ['1001']:
        continue
    
    
    if l[1] == '1001':
        if '早搏' in l[2]:
            num = 1
        elif '心律不齐' in l[2]:
            num = 2
        elif '心动过缓' in l[2]:
            num = 3
        elif '正常心电图' in l[2]:
            num = 4
        else:
            num = 0
        train.loc[train['vid'] == l[0], l[1]] = num
    else:
        continue

test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature.csv', index = False)
test.to_csv('test_feature.csv', index = False)

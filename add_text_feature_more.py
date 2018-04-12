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

    if l[1] not in ['1001', '0118']:
        continue
    
    '''
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
    '''
    if l[1] == '0118':
        if '系统分离' in l[2]:
            num1 = 1
        elif '未见分离' in l[2] or '未见明显分离' in l[2]:
            num1 = 2
        else:
            num1 = 0
            
        if '强回声' in l[2]:
            num2 = 1
        elif '弱回声' in l[2] or '低回声' in l[2]:
            num2 = 2
        elif '无回声' in l[2]:
            num2 = 3
        else:
            num2 = 0
        train.loc[train['vid'] == l[0], l[1] + '_1'] = num1
        train.loc[train['vid'] == l[0], l[1] + '_2'] = num2
    else:
        continue

test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature.csv', index = False)
test.to_csv('test_feature.csv', index = False)

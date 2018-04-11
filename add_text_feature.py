import pandas as pd 

f = open('data.csv', 'r')
lines = f.readlines()[1:]
lines = [l[:-1].split('$') for l in lines]

train = pd.read_csv('train_feature_numerical.csv')
test = pd.read_csv('test_feature_numerical.csv')

len_train = len(train)
print(len(train))

train = train.append(test)
print(len(train))

for i in range(0, len(lines)):
    if i % 100000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]

    if l[1] not in ['2302', '0116', '0113']:
        continue
    
    
    if l[1] == '2302': 
        if l[2] == '健康' or l[2] == ' 健康':
            num = 1
        elif l[2] == '亚健康':
            num = 2
        elif l[2] == '疾病':
            num = 3
        else:
            num = 0
        train.loc[train['vid'] == l[0], l[1]] = num
    elif l[1] == '0116':
        if '强回声' in l[2]:
            num = 1
        elif '弱回声' in l[2]:
            num = 2
        elif '无回声' in l[2]:
            num = 3
        elif '高回声' in l[2]:
            num = 4
        elif '低回声' in l[2]:
            num = 5
        else:
            num = 0
        train.loc[train['vid'] == l[0], l[1]] = num
    elif l[1] == '0113':
        if '强回声' in l[2]:
            num1 = 1
        elif '回声呈点状密集' in l[2]:
            num1 = 2
        elif '回声均匀' in l[2]:
            num1 = 3
        else:
            num1 = 0

        if '欠清晰' in l[2] or '不清晰' in l[2]:
            num2 = 1
        elif '清晰' in l[2]:
            num2 = 2
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

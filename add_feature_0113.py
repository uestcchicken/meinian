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

count1 = [0] * 4
count2 = [0] * 3

for i in range(0, len(lines)):
    if i % 100000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]

    if l[1] not in ['0113']:
        continue
    ###################################################    
    if '强回声' in l[2]:
        num1 = 1
        count1[1] += 1
    elif '回声呈点状密集' in l[2]:
        num1 = 2
        count1[2] += 1
    elif '回声均匀' in l[2]:
        num1 = 3
        count1[3] += 1
    else:
        num1 = 0
        count1[0] += 1
        
    
    
    if '欠清晰' in l[2] or '不清晰' in l[2]:
        num2 = 1
        count2[1] += 1
    elif '清晰' in l[2]:
        num2 = 2
        count2[2] += 1
    else:
        num2 = 0
        count2[0] += 1
    ###################################################
    train.loc[train['vid'] == l[0], l[1] + '_1'] = num1
    train.loc[train['vid'] == l[0], l[1] + '_2'] = num2
    

test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature.csv', index = False)
test.to_csv('test_feature.csv', index = False)

print(count1)
print(count2)
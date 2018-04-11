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

count = [0, 0, 0, 0, 0, 0]

for i in range(0, len(lines)):
    if i % 100000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]

    ###################################################
    if l[1] not in ['0116']:
        continue
        
    if '强回声' in l[2]:
        num = 1
        count[1] += 1
    elif '弱回声' in l[2]:
        num = 2
        count[2] += 1
    elif '无回声' in l[2]:
        num = 3
        count[3] += 1
    elif '高回声' in l[2]:
        num = 4
        count[4] += 1
    elif '低回声' in l[2]:
        num = 5
        count[5] += 1
    else:
        num = 0
        count[0] += 1
    ###################################################
    
    
    
    train.loc[train['vid'] == l[0], l[1]] = num
    

test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature.csv', index = False)
test.to_csv('test_feature.csv', index = False)

print(count)
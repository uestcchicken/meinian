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

count = [0, 0, 0]

for i in range(0, len(lines)):
    if i % 100000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]

    if l[1] not in ['2302']:
        continue
        
    if l[2] == '健康' or l[2] == ' 健康':
        num = 0
        count[0] += 1
    elif l[2] == '亚健康':
        num = 1
        count[1] += 1
    elif l[2] == '疾病':
        num = 2
        count[2] += 1
    else:
        continue
    
    train.loc[train['vid'] == l[0], l[1]] = num
    

test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature.csv', index = False)
test.to_csv('test_feature.csv', index = False)

print(count)
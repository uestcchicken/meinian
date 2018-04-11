import pandas as pd 

f = open('data.csv', 'r')
lines = f.readlines()[1:]
lines = [l[:-1].split('$') for l in lines]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

len_train = len(train)
print(len(train))

train = train.append(test)
print(len(train))

for i in range(0, len(lines)):
    if i % 10000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]
    
    
    ########################
    '''
    if l[1] not in ['2403', '2404', '2405', '0424', '314', '1840', '3193', '1850', '10004', '190', '191', '10002', '10003', '1115', '1117', '1814', '1815', '183', '192', '193', '2174']:
        continue
    '''    
        
        
        
    try:
        #print(l[2][:-1])
        num = float(l[2][:-1])
    except:
        continue

    train.loc[train['vid'] == l[0], l[1]] = num
    

test = train[len_train:]
train = train[:len_train]

train = train.sample(frac = 1)
train.to_csv('train_feature_numerical.csv', index = False)
test.to_csv('test_feature_numerical.csv', index = False)
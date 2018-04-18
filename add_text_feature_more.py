import pandas as pd 
import numpy as np

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

    #if l[1] not in ['1001', '0118', '0437', '0434', '1402', '4001']:
    if l[1] not in ['0409', '3301', '3399', '30007']:
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
    
    if l[1] == '0437':
        if '高' in l[2]:
            num = 1
        elif '未见异常' in l[2] or '正常' in l[2] or '无异常' in l[2] or '无特殊记载' in l[2]:
            num = 2
        elif 'CM' in l[2]:
            num = 3
        else:
            num = 0
        train.loc[train['vid'] == l[0], l[1]] = num 
    
    if l[1] == '0434':
        if '高血压史' in l[2]:
            num1 = 1
        elif '血压偏高' in l[2]:
            num1 = 2
        else:
            num1 = 0
            
        if '糖尿病史' in l[2]:
            num2 = 1
        elif '血糖偏高' in l[2]:
            num2 = 2
        else:
            num2 = 0
            
        if '脂肪肝史' in l[2]:
            num3 = 1
        elif '血脂偏高' in l[2]:
            num3 = 2
        else:
            num3 = 0
            
        if '结石' in l[2]:
            num4 = 1
        else:
            num4 = 0
            
        if '阑尾炎' in l[2]:
            num5 = 1
        else:
            num5 = 0
            
        if '甲肝史' in l[2]:
            num6 = 1
        else:
            num6 = 0
        train.loc[train['vid'] == l[0], l[1] + '_1'] = num1
        train.loc[train['vid'] == l[0], l[1] + '_2'] = num2
        train.loc[train['vid'] == l[0], l[1] + '_3'] = num3
        train.loc[train['vid'] == l[0], l[1] + '_4'] = num4
        train.loc[train['vid'] == l[0], l[1] + '_5'] = num5
        train.loc[train['vid'] == l[0], l[1] + '_6'] = num6
    if l[1] == '1402':
        if '弹性降低' in l[2] or '弹性减退' in l[2]:
            num1 = 1
        else:
            num1 = 0
            
        if '动脉硬化' in l[2]:
            num2 = 1
        else:
            num2 = 0
            
        if '速度减慢' in l[2] or '速度略减慢' in l[2]:
            num3 = 1
        elif '速度增快' in l[2] or '速度略增快' in l[2]:
            num3 = 2
        else:
            num3 = 0
        train.loc[train['vid'] == l[0], l[1] + '_1'] = num1
        train.loc[train['vid'] == l[0], l[1] + '_2'] = num2
        train.loc[train['vid'] == l[0], l[1] + '_3'] = num3
    
    if l[1] == '4001':
        if '轻度减弱' in l[2]:
            num1 = 2
        elif '中度减弱' in l[2]:
            num1 = 3
        elif '重度减弱' in l[2]:
            num1 = 4
        elif '减弱趋势' in l[2]:
            num1 = 1
        else:
            num1 = 0
            
        if '轻度硬化' in l[2]:
            num2 = 1
        elif '硬化稍高' in l[2]:
            num2 = 2
        elif '硬化可能' in l[2]:
            num2 = 3
        elif '相比（稍硬、硬）' in l[2]:
            num2 = 4
        elif '动脉硬化' in l[2]:
            num2 = 5
        else:
            num2 = 0

        train.loc[train['vid'] == l[0], l[1] + '_1'] = num1
        train.loc[train['vid'] == l[0], l[1] + '_2'] = num2
    '''
    if l[1] == '0409':
        if '高血压史' in l[2]:
            num1 = 1
        else:
            num1 = 0        
        if '糖尿病史' in l[2]:
            num2 = 1
        else:
            num2 = 0
        if '脂肪肝史' in l[2]:
            num3 = 1
        else:
            num3 = 0        
        if '血压偏高' in l[2]:
            num4 = 1
        else:
            num4 = 0
        if '血脂偏高' in l[2]:
            num5 = 1
        else:
            num5 = 0        
        if '尿酸偏高' in l[2]:
            num6 = 1
        else:
            num6 = 0
        if '心律不齐' in l[2]:
            num7 = 1
        else:
            num7 = 0        
        if '冠心病史' in l[2]:
            num8 = 1
        else:
            num8 = 0
        if '心动过缓' in l[2]:
            num9 = 1
        else:
            num9 = 0      
        if '早搏' in l[2]:
            num10 = 1
        else:
            num10 = 0     

        train.loc[train['vid'] == l[0], l[1] + '_1'] = num1
        train.loc[train['vid'] == l[0], l[1] + '_2'] = num2
        train.loc[train['vid'] == l[0], l[1] + '_3'] = num3
        train.loc[train['vid'] == l[0], l[1] + '_4'] = num4
        train.loc[train['vid'] == l[0], l[1] + '_5'] = num5
        train.loc[train['vid'] == l[0], l[1] + '_6'] = num6
        train.loc[train['vid'] == l[0], l[1] + '_7'] = num7
        train.loc[train['vid'] == l[0], l[1] + '_8'] = num8
        train.loc[train['vid'] == l[0], l[1] + '_9'] = num9
        train.loc[train['vid'] == l[0], l[1] + '_10'] = num10
    elif l[1] == '3301':
        if '阳性' in l[2]:
            num1 = 1
        elif '阴性' in l[2]:
            num1 = 2
        else:
            num1 = 0
        train.loc[train['vid'] == l[0], l[1]] = num1
    elif l[1] == '3399':
        if '淡黄色' in l[2]:
            num1 = 1
        elif '黄色' in l[2] or 'yellow' in l[2]:
            num1 = 2
        else:
            num1 = 0
        train.loc[train['vid'] == l[0], l[1]] = num1
    elif l[1] == '30007':
        if 'Ⅳ' in l[2] or 'IV' in l[2] or 'iv' in l[2]:
            num1 = 4
        elif 'Ⅲ' in l[2] or 'III' in l[2] or 'iii' in l[2]:
            num1 = 3
        elif 'Ⅱ' in l[2] or 'II' in l[2] or 'ii' in l[2]:
            num1 = 2
        elif 'Ⅰ' in l[2] or 'I' in l[2] or 'i' in l[2]:
            num1 = 1
        else:
            num1 = 0
        train.loc[train['vid'] == l[0], l[1]] = num1
    else:
        continue

test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature.csv', index = False)
test.to_csv('test_feature.csv', index = False)

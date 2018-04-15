1. 拼接两个data文件，仅一个头行

对train.csv手动删除或修改了一些不能转换为float的值

2. gen_feature_numerical.py  
生成二维数据（仅取数值型）feature_numerical.csv

3. add_text_feature.py(add_text_feature_more.py)  
加入特征2302,0116,0113,(1001)

4. cut_lose.py  
丢弃缺失值比例较大的列得到cutlose.csv

5. cut_useless.py(合并至4)  
删除异常数据，仅对train，得到train_cut_useless.csv

对test_cutlose.csv将32b7cddb800f4218e77ec9e4d9092fa5的-90手动改为均值58

6. train.py  
train_cut_useless.csv和test_cutlose.csv

其他：

- gen_texts.py  
生成最多的几个文字型信息

todo：

- data里哪些是1人1项多次的

## v1.0

#### add_text_feature.py

```python
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
```

#### cut_lose.py

```python
if values[True] > 47500:
      drop_columns.append(name)
```

#### cut_useless.py

```python
f = f[f['A'] > 0.0]
f = f[f['B'] > 0.0]
f = f[f['E'] > 0.0]
f = f[f['2403'] < 5000]
f = f[f['10004'] >= 0.0]
f = f[f['1814'] < 1800]
```

#### train.py

```python
categorical = ['2302', '0116', '0113_1', '0113_2']

params = {
    'boosting': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'application': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 16,
    'nthread': 8
}
```

#### result

valid: 0.0304  
score: 0.0326

## v1.1

- 加入feature1001
- 使用自定义fobj

#### add_text_feature.py

```python
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
```

#### train.py

```python
def obj_function(preds, train_data):
    y = train_data.get_label()
    p = preds
    grad = 2.0 / (p + 1.0) * (np.log1p(p) - np.log1p(y))
    hess = 2.0 / np.square(p + 1.0) * (1.0 - np.log1p(p) + np.log1p(y))
    return grad, hess
    
def eval_function(preds, train_data):
    labels = train_data.get_label()
    return 'loss', np.mean(np.square(np.log1p(preds) - np.log1p(labels))), False

categorical = ['2302', '0116', '0113_1', '0113_2', '1001']
    
params = {
    'boosting': 'gbdt',
    'objective': 'none',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'nthread': 8
}
```

#### result 

valid: 0.0284  
score: 0.0311

## v1.2

- 加入feature0118_1, 0118_2

#### add_text_feature.py

```python
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
```

#### result 

valid: 0.0284  
score: 0.0310

## v1.3

- cut_lose.py不删除缺失值较多的列

#### result

valid: 0.0282

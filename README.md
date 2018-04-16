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

- 1402
- 1403
- data里哪些是1人1项多次的

## v1.0

valid: 0.0304  
score: 0.0326

## v1.1

- 加入feature1001
- 使用自定义fobj

valid: 0.0284  
score: 0.0311

## v1.2

- 加入feature0118_1, 0118_2

valid: 0.0284  
score: 0.0310

## v1.3

- cut_lose.py不删除缺失值较多的列
- 加入feature0437
- 重新随机feature.csv

valid: 0.0290  
score: 0.0308

## v1.4

- 加入feature0434_1~6
- 穷举法瞎几把调参

valid: 0.0284  
score: 0.0303


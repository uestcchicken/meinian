1. 拼接两个data文件，仅一个头行

2. gen_feature_numerical.py  
生成二维数据（仅取数值型）feature_numerical.csv

3. add_text_feature.py  
加入特征2302,0116,0113

4. gen_cutlose.py  
丢弃缺失值比例较大的列得到cutlose.py

5. cut_useless.py  
删除异常数据，仅对train，得到train_cut_useless.csv

6. train.py  
train_cut_useless.csv和test_cutlose.csv

其他：

- gen_texts.py  
生成最多的几个文字型信息

todo：

- data里哪些是1人1项多次的
- fobj
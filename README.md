- gen_feature_numerical.py  
生成二维数据（仅取数值型）feature_numerical.csv

- add_feature_2302.py  
将2302项加入feature_numerical.csv得到feature.csv

- gen_cutlose.py  
丢弃缺失值比例较大的列得到cutlose.py

- cut_useless.py  
删除异常数据，仅对train，得到train_cut_useless.csv

- train.py  
train_cut_useless.csv和test_cutlose.csv





其他：

- gen_texts.py  
生成最多的几个文字型信息


import jieba
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
import gc



f = open('texts_done/1001.csv')
lines = f.readlines()
f.close()

words_all = []

print(len(lines))

for l in lines:
    #seg = jieba.cut(l[:-1], cut_all = True)
    seg = jieba.cut_for_search(l[:-1])
    words = ' '.join(seg)
    
    #print(words)
    words_all.append(words)
    
vectorizer = CountVectorizer()
transformer = TfidfTransformer()

tf = vectorizer.fit_transform(words_all).toarray()

tfidf = transformer.fit_transform(tf).toarray()
print(tfidf[0])
print(len(tfidf))

#print(tf[0])
#print(len(tf[0]))
names = vectorizer.get_feature_names()
print(names)
print(len(names))
print(len(tf[0]))

pd_data = pd.DataFrame(tfidf, columns = names)
print(pd_data.info())
print(pd_data.sample(5))

drop = []
remain = []
for c in list(pd_data.columns):
    values = (pd_data[c] != 0).value_counts()
    if values[True] > 1000:
        remain.append(c)
    else:
        drop.append(c)

print(remain)


'''


f = open('data.csv', 'r')
lines = f.readlines()[1:]
lines = [l[:-1].split('$') for l in lines]
f.close()

train = pd.read_csv('train_feature.csv')
test = pd.read_csv('test_feature.csv')

len_train = len(train)
print(len(train))

train = train.append(test)
print(len(train))

del test 
gc.collect()

counter = 0

for i in range(0, len(lines)):
    if i % 100000 == 0:
        print(i, '/', len(lines), i / len(lines) * 100 , '%')
    l = lines[i]

    if l[1] != '1001':
        continue
    
    features = tfidf[counter]
    for j in range(247):
        train.loc[train['vid'] == l[0], l[1] + '_' + names[j]] = features[j]
    counter += 1
    print(counter)
        
test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature_jiebatest.csv', index = False)
test.to_csv('test_feature_jiebatest.csv', index = False)
'''
import jieba
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
import gc

typenames = ['1001', '0113', '0116', '0118', '2302', '0114', '0115', '0117', '1308']


train = pd.read_csv('train_feature.csv')
test = pd.read_csv('test_feature.csv')

len_train = len(train)
print(len(train))
train = train.append(test)
print(len(train))

del test 
gc.collect()

for typename in typenames:
    print('###################start', typename, '...')
    f = open('texts_vid/' + typename + '.csv')
    lines = f.readlines()
    lines = [l[:-1].split(',') for l in lines]
    #print(lines[0])
    lines_vid = [l[0] for l in lines]
    lines_content = [l[1] for l in lines]
    f.close()
    
    del f 
    del lines
    gc.collect()

    words_all = []
    
    print('cutting with jieba...')
    for l in lines_content:
        #seg = jieba.cut(l[:-1], cut_all = True)
        seg = jieba.cut_for_search(l[:-1])
        words = ' '.join(seg)
        #print(words)
        words_all.append(words)
    
    del lines_content
    gc.collect()
        
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    
    print('generating tfidf...')
    tf = vectorizer.fit_transform(words_all).toarray()
    tfidf = transformer.fit_transform(tf).toarray()
    #print(tfidf[0])
    #print(len(tfidf))

    print('blablablabla...')
    names = vectorizer.get_feature_names()
    for i in range(len(names)):
        names[i] = typename + '_' + names[i]

    #print(names)
    #print(len(names))
    #print(len(tfidf[0]))

    vid_tfidf = [[l] for l in lines_vid]
    
    del lines_vid
    gc.collect()
    
    for i in range(len(vid_tfidf)):
        vid_tfidf[i].extend(tfidf[i])
    #print(vid_tdidf[0])
    vid_names = ['vid']
    vid_names.extend(names)
    #print(vid_names)

    
    pd_data = pd.DataFrame(vid_tfidf, columns = vid_names)
    #print(pd_data.info())
    #print(pd_data.sample(5))
    del vid_tfidf
    del vid_names
    gc.collect()
    
    print('dropping too few...')
    drop = []
    remain = []
    for c in list(pd_data.columns):
        values = (pd_data[c] != 0).value_counts()
        if values[True] > 200:
            remain.append(c)
        else:
            drop.append(c)

    #print(remain)

    pd_data = pd_data[remain]
    #print(pd_data.info())
    #print(pd_data.sample(5))
    print('merging into train test...')
    train = train.merge(pd_data, on = ['vid'], how = 'left')
    #print(train.info())
    #print(train.sample(5))
    
    del pd_data
    gc.collect()


test = train[len_train:]
train = train[:len_train]

train.to_csv('train_feature_jiebatest.csv', index = False)
test.to_csv('test_feature_jiebatest.csv', index = False)

import pandas as pd 

data = pd.read_csv('data.csv', sep = '$')
print(data['table_id'].value_counts())






#names = ['0102', '0101', '1815', '2302']
names = ['2302']

for name in names:
    data_n = data[data['table_id'] == name]
    print(data_n['field_results'].value_counts())
    #print(data.info())
    #print(data.sample(10))

    data_n = data_n['field_results']
    data_n.to_csv('texts/' + name + '.csv', index = False, header = False)

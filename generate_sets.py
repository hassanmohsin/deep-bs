import pandas as pd

datafile = "./dataset/refined-set/index/INDEX_refined_data.2016"
df = pd.read_csv(datafile, skiprows=6, sep='\s+', header=None, usecols=[0, 3, 4], names=['code', 'affinity', 'Kd/Ki'])
df['afftype'] = df['Kd/Ki'].apply(lambda x: x.split('=')[0])
refined_ids = open("./data/refined_ids.txt", 'r').readline().split(',')
refined_df = df[df.code.isin(refined_ids)]
core_df = df[~df.code.isin(refined_ids)]
refined_df.to_csv('./data/refined_set.csv', index=None)
core_df.to_csv('./data/core_set.csv', index=None)

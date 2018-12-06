import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("shuffled-full-set-hashed.csv",header=None)
df.columns = [['class','text']]
df.head()
df['class'].value_counts()

df['text'].isnull().value_counts()
df = df.dropna(axis=0, subset=['text'])

df['class'].value_counts()

rus=RandomUnderSampler(random_state=0,return_indices=True)

Xarray=df['text'].values.astype(str)

features_sampled,labels_sampled,indices=rus.fit_sample(Xarray.reshape(-1,1),df['class'])

output_array=np.concatenate((labels_sampled,features_sampled),axis=1)

df.to_csv('sampled_with_names.csv')

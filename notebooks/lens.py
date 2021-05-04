#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import pickle

import lenskit
from lenskit.batch import MultiEval
from lenskit.crossfold import partition_users, SampleN
from lenskit.algorithms import basic, als
from lenskit import topn, util

import pandas as pd
import numpy as np
from scipy import stats
import binpickle
import matplotlib.pyplot as plt


from tqdm.notebook import tqdm_notebook as tqdm
import argparse




def main(train_dir, test_dir):
    df_train = pd.read_parquet(train_dir)
    df_test = pd.read_parquet(test_dir)
    df_train.rename(columns = {'user_id':'user', 'track_id':'item', 'count':'rating'}, inplace = True)
    df_test.rename(columns = {'user_id':'user', 'track_id':'item', 'count':'rating'}, inplace = True)
    
    
    eval = MultiEval('my-eval', recommend=20)
    
    pairs = list(partition_users(df_train, 5, SampleN(5)))
    eval.add_datasets(pairs, name='Song')
    
    ALS = als.ImplicitMF(iterations=20, reg=0.1, weight=40, method='cg')
    eval.add_algorithms([ALS], attrs=['features'], name='ImplicitMF')
    eval.run(progress=tqdm)
    
    runs = pd.read_csv('my-eval/runs.csv')
    runs.set_index('RunId', inplace=True)
    runs.head()
    
    
    recs = pd.read_parquet('my-eval/recommendations.parquet')
    recs.head()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='popularity baseline model bias.py test')
    parser.add_argument('-tr', '--train', type=str, dest='train_dir', metavar='',
                        default=None, help='Train data directory')
    parser.add_argument('-te', '--test', type=str, dest='test_dir', metavar='',
                        default=None, help='Test data directory')

    args = parser.parse_args()
    
    
    
    main(train_dir=args.train_dir, test_dir=args.test_dir)


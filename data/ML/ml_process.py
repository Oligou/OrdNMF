#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert

Thanks to Dawen Liang
"""

import numpy as np
import cPickle as pickle
import scipy.sparse as sparse
import pandas as pd

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import os
import scipy.sparse as sparse
import cPickle as pickle
import sqlite3

data_file = 'ml-20m/ratings.csv'
movie_file = 'ml-20m/movies.csv'

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc, min_sc):
    songcount = get_count(tp, 'sid')
    tp = tp[tp['sid'].isin(songcount.index[songcount >= min_sc])]
    
    usercount = get_count(tp, 'uid')
    tp = tp[tp['uid'].isin(usercount.index[usercount >= min_uc])]
    
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid') 
    return tp, usercount, songcount

def make_csr(tp,shape,row_index,col_index):
    row,col = (np.array(tp[row_index]),np.array(tp[col_index]))
    data = np.array(tp['count'])
    return sparse.csr_matrix((data,(row,col)), shape=shape)

#%%
def process(seed,
            min_user_count, min_song_count,
            U=None, I=None):

    saved_args = locals()
    np.random.seed(seed)
    
    ###########################
    ###########################
    #%% TASTE PROFILE DATASET
    ###########################
    ###########################
    
    tp_original = pd.read_table(data_file, sep=',')
    tp_original.columns = ['uid', 'sid','count','time']
    tp = tp_original.copy()
    tp['count'] = 2*tp['count']
    tp['count'] = tp['count'].astype(int)
    
    tp, usercount, songcount = filter_triplets(tp, min_user_count,min_song_count)
    
    #############################
    #%% SELECT SUBSET USER
    if U is not None:
        unique_user = usercount.index
        p_users = usercount / usercount.sum()
        select_user = np.random.choice(unique_user, size=U, replace=False, p=p_users.tolist())
        select_user = pd.DataFrame(select_user,columns=['uid'])
        tp = tp.merge(select_user,on='uid')
    
    if I is not None:
        unique_song = songcount.index
        p_songs = songcount / songcount.sum()
        select_song = np.random.choice(unique_song, size=I, replace=False, p=p_songs.tolist())
        select_song = pd.DataFrame(select_song,columns=['sid'])
        tp = tp.merge(select_song,on='sid')
    
    if U is not None or I is not None:
        tp, usercount, songcount = filter_triplets(tp, min_user_count,min_song_count)
    
    #########################
    #########################
    #%% CREATE MATRICES
    #########################
    #########################
    
    unique_user = tp.uid.unique()
    U = len(unique_user)
    user = pd.DataFrame({'uid':unique_user,'user_index': range(U)})
    
    unique_song = tp.sid.unique()
    I = len(unique_song)
    song = pd.DataFrame({'sid':unique_song,'song_index': range(I)})
    
    tp = tp.merge(user,on='uid')
    tp = tp.merge(song,on='sid')
    
    Y_listen = make_csr(tp,(U,I),'user_index','song_index')
    
    #########################
    #########################
    #%% METADATA
    #########################
    #########################
    metadata_original = pd.read_table(movie_file, sep=',')
    
    movies_metadata = song.merge(metadata_original, left_on='sid', right_on='movieId')
    
    #########################
    #########################
    #%% SAVE
    #########################
    #########################
    filename = 'ml_' + str(seed) + \
                '_U%.2e'%U + '_I%.2e'%I + \
                '_min_uc%d_sc%d' % (min_user_count,min_song_count)
    with open(filename, 'wb') as handle:
        pickle.dump({'Y_listen':Y_listen,'movie':song, 'movies_metadata':movies_metadata,
                     'user':user,'input':saved_args}, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%    
if True:
    process(seed=145,min_user_count=20, min_song_count=20,
                    U=20000, I=None)

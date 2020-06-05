#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ogouvert

Thanks to Dawen Liang
"""

# data available at http://millionsongdataset.com/tasteprofile/

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import os
import scipy.sparse as sparse
import cPickle as pickle
import sqlite3
    
TPS_DIR = 'TPS'
TP_file = os.path.join(TPS_DIR, 'train_triplets.txt')
md_dbfile = os.path.join(TPS_DIR, 'track_metadata.db')

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
def tps_process(seed,
                min_user_count, min_song_count,
                U=None, I=None):

    saved_args = locals()
    np.random.seed(seed)
    
    ###########################
    ###########################
    #%% TASTE PROFILE DATASET
    ###########################
    ###########################
    
    tp_original = pd.read_table(TP_file, header=None, names=['uid', 'sid', 'count'])
    tp = tp_original.copy()
    
    tp, usercount, songcount = filter_triplets(tp, min_user_count,min_song_count)
    
    #############################
    #%% SELECT SUBSET USER
    if U is not None:
        unique_user = usercount.index
        p_users = usercount / usercount.sum()
        U = min(U,len(unique_user))
        select_user = np.random.choice(unique_user, size=U, replace=False, p=p_users.tolist())
        select_user = pd.DataFrame(select_user,columns=['uid'])
        tp = tp.merge(select_user,on='uid')
    
    if I is not None:
        unique_song = songcount.index
        p_songs = songcount / songcount.sum()
        I = min(I,len(unique_song))
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
#    Y_listen_a = Y_listen.A
    
    #%% Filtering ok?
    # user
    print Y_listen.shape
    print 'nb user en trop' + str(np.sum((Y_listen>0).sum(1)<min_user_count))
    print 'nb item en trop' + str(np.sum((Y_listen>0).sum(0)<min_song_count))
    
    #########################
    #########################
    #%% METADATA
    #########################
    #########################
    conn = sqlite3.connect(md_dbfile)
    #res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #for name in res:
    #    print name[0]
    
    songs_metadata = pd.read_sql_query("SELECT * FROM songs",conn)
    songs_metadata = songs_metadata.drop_duplicates(['song_id'])
    songs_metadata = song.merge(songs_metadata, left_on='sid', right_on='song_id')
    
    #########################
    #########################
    #%% SAVE
    #########################
    #########################
    filename = 'tps_' + str(seed) + \
                '_U%.2e'%U + '_I%.2e'%I + \
                '_min_uc%d_sc%d' % (min_user_count,min_song_count)
    with open(filename, 'wb') as handle:
        pickle.dump({'Y_listen':Y_listen,'song':song,'songs_metadata':songs_metadata,
                     'user':user,'input':saved_args}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
#%%
if True:
    tps_process(seed=12345,min_user_count=20, min_song_count=20,
                    U=20000, I=20000)
    
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import re
import warnings
import random

import numpy as np
import scipy as sp
import pandas as pd

from numba import jit
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[39]:


song_meta = pd.read_json('song_meta.json', typ = 'frame', encoding='utf-8')


# # train data로 songs_y, tags_y vs support, confidence, lift 모델 학습

# - train.songs_X와 train.tags_X의 길이가 0인 경우는 없음

# In[51]:


train = pd.read_json('train.json', typ = 'frame', encoding='utf-8')
val = pd.read_json('val.json', typ = 'frame', encoding='utf-8')
test = pd.read_json('test.json', typ = 'frame', encoding='utf-8')
train = pd.concat([train, val, test])


# In[11]:


song_co_occurrence_csr = sp.sparse.load_npz("all_song_co_occurrence_coo.npz").tocsr()
tag_co_occurrence_csr = sp.sparse.load_npz("all_tag_co_occurrence_coo.npz").tocsr()
tag_tag_co_occurrence_csr = sp.sparse.load_npz("all_tag_tag_co_occurrence_coo.npz").tocsr()


# In[52]:


train_song_count = {song:0 for song in range(len(song_meta))}


# In[53]:


for songs in train.songs:
    for song in songs:
        train_song_count[song] += 1


# In[54]:


train_song_count = np.array([train_song_count[i] for i in range(len(song_meta))])
del song_meta


# In[56]:


tags_dict = {tag:i for i, tag in enumerate(train.tags.explode().unique())}
index_tags= {index:tag for tag, index in tags_dict.items()}


# # validation 데이터 예측

# ### validation song 예측

# In[58]:


# 비어 있는 playlist에 대해서는 등장횟수 상위 100song으로 예측
train_songs_co_occurrence_rank = np.array(song_co_occurrence_csr.sum(axis=1))[:,0].argsort()[::-1]
train_songs_co_occurrence_rank
np.array(song_co_occurrence_csr.sum(axis=1))[train_songs_co_occurrence_rank]


# In[78]:


test = pd.read_json('test.json', typ = 'frame', encoding='utf-8')


# In[ ]:


warnings.filterwarnings(action='ignore')
song_test_pred = []
for i, songs in enumerate(test.songs):
    if i > 0 and i % 1000 == 0:
        print(i, "th completed", sep="")
    
    song_i_pred = []
    if not songs:
        tags = test.tags[i]
        if tags:
            tag_index = list(map(lambda x : tags_dict[x], tags))
            song_i_pred = np.array(tag_co_occurrence_csr[:,tag_index].sum(axis=1))[:,0].argsort()[:-101:-1]
        else:
            song_test_pred += [train_songs_co_occurrence_rank[:100].tolist()]
        continue
    
    
    song_i_candidate = np.array(song_co_occurrence_csr[songs,:].sum(axis=0))[0,:].argsort()[::-1]
    
    count = 0
    for song in song_i_candidate:
        if not song in songs:
            song_i_pred += [song]
            count += 1
            if count == 100: break
    if count < 100:
        for song in train_songs_co_occurrence_rank:
            if not song in songs:
                song_i_pred += [song]
                count += 1
                if count == 100: break
    song_test_pred += [song_i_pred]
warnings.filterwarnings(action='default')


# In[60]:


train_tags_co_occurrence_rank = list(map(lambda x: index_tags[x], np.array(tag_co_occurrence_csr.sum(axis=0))[0,:].argsort()[::-1]))
print(train_tags_co_occurrence_rank[:10])
np.array(tag_co_occurrence_csr.sum(axis=0))[0,:][np.array(tag_co_occurrence_csr.sum(axis=0))[0,:].argsort()[::-1]]


# In[ ]:


warnings.filterwarnings(action='ignore')
tag_test_pred = []
tag_top10 = [train_tags_co_occurrence_rank[:10]]
for i, songs in enumerate(test.songs):
    if i > 0 and i % 1000 == 0:
        print(i, "th completed", sep="")
    
    tag_i_pred = []
    if not songs:
        if tags:
            tag_index = list(map(lambda x : tags_dict[x], tags))
            tag_i_pred = np.array(tag_tag_co_occurrence_csr[tag_index,:].sum(axis=0))[0,:].argsort()[:-11:-1]
        
        else:
            tag_test_pred += tag_top10
        continue
    
    tag_i_candidate = np.array(tag_co_occurrence_csr[songs,:].sum(axis=0))[0,:].argsort()[::-1]
    
    tags = test.tags[i]
    count = 0
    for tag_index in tag_i_candidate:
        tag = index_tags[tag_index]
        if not tag in tags:
            tag_i_pred += [tag]
            count += 1
            if count == 10: break
    

    tag_test_pred += [tag_i_pred]
warnings.filterwarnings(action='default')


# In[64]:


test_result = [{'id':test.id[i], 'songs':song_test_pred[i], 'tags':tag_test_pred[i]} for i in range(test.shape[0])]


# In[ ]:


test_result = re.sub("\'", '\"', str(test_result))


# In[ ]:


with open('results.json', 'w', encoding='utf-8') as f:
    f.write(str(test_result))

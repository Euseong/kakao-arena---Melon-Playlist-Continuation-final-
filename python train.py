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


# In[106]:


song_meta = pd.read_json('song_meta.json', typ = 'frame', encoding='utf-8')
train = pd.read_json('train.json', typ = 'frame', encoding='utf-8')
val = pd.read_json('val.json', typ = 'frame', encoding='utf-8')
test = pd.read_json('test.json', typ = 'frame', encoding='utf-8')
train = pd.concat([train, val, test])


# In[8]:


# co-occurrence matrix for songs
num_songs = song_meta.shape[0]
def cooccur(song_lists):
    import scipy as sp
    i = 0
    result = sp.sparse.dok_matrix((num_songs,num_songs), dtype='int32')
    for songs in song_lists:
        if i > 9999 and i % 10000 == 0:
            print(i, "th completed", sep="")
        i = 0
        for song in songs:
            for co_occured_song in songs[(i+1):]:
                result[song, co_occured_song] += 1
                result[co_occured_song, song] += 1
            i += 1
    return result


# In[76]:


# train.songs 이용 2~3시간 걸림
song_co_occurrence = cooccur(train.songs)
sp.sparse.save_npz("all_song_co_occurrence.npz", song_co_occurrence.tocoo())


# In[ ]:


song_co_occurrence_csr = song_co_occurrence.tocsr()
del song_co_occurrence


# - 특정 song이 포함될 conditional probability를 계산하기 위한 song 별 총 등장횟수 계산

# In[107]:


train_song_count = {song:0 for song in range(len(song_meta))}


# In[108]:


for songs in train.songs:
    for song in songs:
        train_song_count[song] += 1


# In[109]:


train_song_count = np.array([train_song_count[i] for i in range(len(song_meta))])


# In[28]:


# 태그 unique 값 추출
tags_dict = {tag:i for i, tag in enumerate(train.tags.explode().unique())}
index_tags = {index:tag for tag, index in tags_dict.items()}


# In[81]:


# co-occurrence matrix for tags with respect to songs
def cooccur_tag(song_lists, tag_lists, n_songs, n_tags):
    import scipy as sp
    result = sp.sparse.dok_matrix((n_songs, n_tags), dtype='int16')
    i = 0
    for songs, tags in zip(song_lists, tag_lists):
        if i > 9999 and not i % 10000:
            print(i, "th list completed", sep="")
        songs_in_list = len(songs)
        for song in songs:
            for tag in tags:
                result[song, tags_dict[tag]] += 1
        i += 1
    return result


# In[82]:


tag_co_occurrence = cooccur_tag(train.songs, train.tags, n_songs=num_songs, n_tags=len(tags_dict))
tag_co_occurrence = tag_co_occurrence.tocoo()
sp.sparse.save_npz('all_tag_co_occurrence_coo.npz', tag_co_occurrence)
tag_co_occurrence_csr = tag_co_occurrence.tocsr()
del tag_co_occurrence


# In[30]:


train_tag_index_count = {index:0 for index in range(len(tags_dict))}
train_tag_index_count = np.array([train_tag_index_count[i] for i in range(len(tags_dict))])


# In[31]:


for tags in train.tags:
    for tag in tags:
        train_tag_index_count[tags_dict[tag]] += 1


# In[32]:


train_tag_index_count


# In[ ]:


# co-occurrence matrix for tags
num_tags = len(tags_dict)
tag_tag_co_occurrence = sp.sparse.dok_matrix((num_tags,num_tags), dtype='int32')
for tags in train.tags:
    i = 0
    if i > 9999 and i % 10000: print(i, "th completed", sep="")
    for tag in tags:
        for co_occurred_tag in tags[(i+1):]:
            tag_tag_co_occurrence[tags_dict[tag], tags_dict[co_occurred_tag]] += 1
            tag_tag_co_occurrence[tags_dict[co_occurred_tag], tags_dict[tag]] += 1
        i += 1


# In[ ]:


sp.sparse.save_npz("all_tag_tag_co_occurrence_coo.npz", tag_tag_co_occurrence.tocoo())


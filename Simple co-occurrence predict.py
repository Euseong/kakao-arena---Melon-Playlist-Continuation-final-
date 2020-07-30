#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re

import numpy as np
import scipy as sp
import pandas as pd

# from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from konlpy.tag import Twitter
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# In[2]:


font_path = 'font/NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
plt.rc('font', family=font_name, size=12)
plt.rcParams["figure.figsize"] = (20, 10)
register_matplotlib_converters()

mpl.font_manager._rebuild()
mpl.pyplot.rc('font', family='NanumGothic')


# In[3]:


pd.options.mode.chained_assignment = None


# In[4]:


genre_gn_all = pd.read_json('genre_gn_all.json', typ='series', encoding='utf-8')
# 장르코드 : gnr_code, 장르명 : gnr_name
genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})
song_meta = pd.read_json('song_meta.json', typ = 'frame', encoding='utf-8')
train = pd.read_json('train.json', typ = 'frame', encoding='utf-8')


# ## 노래들 간 co-occurrence를 이용하여 누락된 노래 100개 예측

# In[9]:


# co-occurrence matrix for songs
num_songs = song_meta.shape[0]
def cooccur(song_lists):
    import scipy as sp
    result = sp.sparse.dok_matrix((num_songs,num_songs), dtype='int32')
    for songs in song_lists:
        i = 0
        for song in songs:
            for co_occured_song in songs[(i+1):]:
                result[song, co_occured_song] += 1
                result[co_occured_song, song] += 1
            i += 1
    return result


# In[55]:


# 2시간 걸림
song_co_occurrence = cooccur(train.songs)


# In[7]:


# song_co_occurrence_coo = song_co_occurrence.tocoo()
# sp.sparse.save_npz("train_song_co_occurrence.npz", song_co_occurrence_coo)


# In[95]:


song_co_occurrence_csr = sp.sparse.load_npz("train_song_co_occurrence.npz").tocsr()


# In[8]:


song_co_occurrence_csr


# In[99]:


# csr(compressed sparse row) matrix가 연산 더 빠름
song_id = 1000
print("nonzero occurrence :", song_co_occurrence_csr[song_id, song_co_occurrence_csr[song_id,:].nonzero()[1]].todense())
print("number of co-occurred songs :", len(song_co_occurrence_csr[song_id,:].nonzero()[1]))


# In[155]:


song_co_occurrence_csr[song_id,:].shape


# In[118]:


song_co_occurrence_csr[song_id,:].todense()


# In[140]:


ith_song = song_co_occurrence_csr[song_id,:].toarray()[0]
ith_song[ith_song.argsort()[::-1]]


# In[141]:


ith_song.argsort()[::-1]


# In[153]:


song_513309 = song_co_occurrence_csr[513309,:].toarray()[0]
print(song_513309.argsort()[::-1])
print(song_513309[song_513309.argsort()[:(-100):-1]]) # 이거 다 더해서 예측


# In[180]:


song_513309[1000]


# In[191]:





# In[194]:





# In[ ]:





# In[186]:


a[:5, 1].shape


# ## 노래들과 태그의 co-occurrence를 이용하여 태그 10개 예측

# In[10]:


# 태그 unique 값 추출
tags_dict = {tag:i for i, tag in enumerate(train.tags.explode().unique())}
index_tags = {index:tag for tag, index in tags_dict.items()}


# In[364]:


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


# In[11]:


train.head()


# In[365]:


tag_co_occurrence = cooccur_tag(train.songs, train.tags, n_songs=num_songs, n_tags=len(tags_dict))
tag_co_occurrence_coo = tag_co_occurrence.tocoo()
# sp.sparse.save_npz('train_tag_co_occurrence_coo.npz', tag_co_occurrence_coo)


# # validation 데이터 예측

# In[12]:


tag_co_occurrence_coo = sp.sparse.load_npz("train_tag_co_occurrence_coo.npz")
tag_co_occurrence_csr = tag_co_occurrence_coo.tocsr()


# In[93]:


tag_co_occurrence_csr


# In[361]:


song_co_occurrence_csr


# In[13]:


val = pd.read_json('val.json', typ = 'frame', encoding='utf-8')


# In[14]:


print(val.shape)
val.head()


# In[387]:


sum([1 if n_songs==0 else 0 for n_songs in list(map(lambda x: len(x), val.songs))])


# #### validation 데이터에서  songs가 비어 있는 playlist는 4379개 (전체 playlist는 23015개)

# In[17]:


songs_0 = val.songs[0]
song_co_occurrence_csr[songs_0, songs_0].diagonal()


# In[18]:


songs_co_occur_0 = sp.sparse.csr_matrix((1, num_songs))
print("max co-occurrence for each song in songs_0")
for song in songs_0:
    print(song_co_occurrence_csr[song,:].max())
    songs_co_occur_0 += song_co_occurrence_csr[song,:]
songs_co_occur_0


# In[420]:


songs_co_occur_0[0, songs_co_occur_0.nonzero()[1]].max()


# In[429]:


print(sorted(np.array(songs_co_occur_0[0, songs_co_occur_0.nonzero()[1]].todense())[0], reverse=True)[:100])


# In[38]:


song_0_pred = np.array(songs_co_occur_0.todense())[0].argsort()[::-1]
songs_co_occur_0[0, song_0_pred].todense()


# In[42]:


num_in_song_0 = 0
for i, song in enumerate(song_0_pred[:100]):
    if song in songs_0:
        num_in_song_0 += 1
        print("index :", i, "// song id :", song)
print(num_in_song_0)


# ### validation song 예측

# In[116]:


train_songs_co_occurrence_rank = np.array(song_co_occurrence_csr.sum(axis=1))[:,0].argsort()[::-1]
train_songs_co_occurrence_rank
np.array(song_co_occurrence_csr.sum(axis=1))[train_songs_co_occurrence_rank]


# In[204]:


song_val_pred = []
for i, songs in enumerate(val.songs):
    if i > 0 and i % 1000 == 0:
        print(i, "th completed", sep="")
    
    if not songs:
        song_val_pred += [train_songs_co_occurrence_rank[:100].tolist()]
        continue
    
    song_i_pred = []
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
    song_val_pred += [song_i_pred]


# In[203]:


print(song_val_pred[1])


# ### validation tag 예측

# In[125]:


tag_co_occurrence_csr


# In[135]:


train_tags_co_occurrence_rank = list(map(lambda x: index_tags[x], np.array(tag_co_occurrence_csr.sum(axis=0))[0,:].argsort()[::-1]))
print(train_tags_co_occurrence_rank[:10])
np.array(tag_co_occurrence_csr.sum(axis=0))[0,:][np.array(tag_co_occurrence_csr.sum(axis=0))[0,:].argsort()[::-1]]


# In[100]:


print(index_tags)


# In[205]:


tag_val_pred = []
tag_top10 = [train_tags_co_occurrence_rank[:10]]
for i, songs in enumerate(val.songs):
    if i > 0 and i % 1000 == 0:
        print(i, "th completed", sep="")
    
    if not songs:
        tag_val_pred += tag_top10
        continue
    
    tag_i_pred = []
    tag_i_candidate = np.array(tag_co_occurrence_csr[songs,:].sum(axis=0))[0,:].argsort()[::-1]
    
    tags = val.tags[i]
    count = 0
    for tag_index in tag_i_candidate:
        tag = index_tags[tag_index]
        if not tag in tags:
            tag_i_pred += [tag]
            count += 1
            if count == 10: break

    tag_val_pred += [tag_i_pred]


# In[154]:


tag_i_pred


# In[162]:


print(np.array(song_val_pred).shape)


# In[211]:


np.array(tag_val_pred).shape


# In[227]:


np.array(song_val_pred[i])


# In[230]:


val_result = [{'id':val.id[i], 'songs':song_val_pred[i], 'tags':tag_val_pred[i]} for i in range(val.shape[0])]


# In[231]:


len(val_result)


# In[232]:


with open('results.json', 'w', encoding='utf-8') as f:
    f.write(str(val_result))
#     for pred in val_result:
        


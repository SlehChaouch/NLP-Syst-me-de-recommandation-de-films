#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
d1 = pd.read_csv('C:\\Users\\HP\\Desktop\\3AMIndS\\data mining\\NLP\\credits.csv')
d2 =pd.read_csv('C:\\Users\\HP\\Desktop\\3AMIndS\\data mining\\NLP\\movies_metadata.csv')
d2 = d2[d2.id!='1997-08-20']  #supprimer les valeurs redondants
d2 = d2[d2.id!='2012-09-29']
d2 = d2[d2.id!='2014-01-01']
d1.id
d2['id'] = d2['id'].astype(int) # Conversion de la colonne "id" en type entier 
d2=d2.merge(d1, on='id') # mergement de deux bases de données

# création d'un systéme qui peut récommander un film à l'aide de colonne "overview" (description de film).
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english') # pour supprimez tous les mots d'arrêt anglais
d2['overview'] = d2['overview'].fillna('') 

from sklearn.metrics.pairwise import linear_kernel
import random
ran = random.randint(1500, 2000)
d3 = d2.head(ran)
tfidf_matrix = tfidf.fit_transform(d3['overview'])
cosine_sim =  linear_kernel(tfidf_matrix, tfidf_matrix, True)
indices = pd.Series(d3.index, index=d3['title']).drop_duplicates()
#création d'une fonction de systeme de recommandation
def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title] 
        sim_scores = list(enumerate(cosine_sim[idx])) # Obtenez les scores de similarité par paire de tous les films avec ce film.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 
        sim_scores = sim_scores[1:11] # Obtenez les scores des 10 films les plus similaires
        movie_indices = [i[0] for i in sim_scores] 
        return d3['title'].iloc[movie_indices] # Retourner les 10 films les plus similaires
# tester le code 
get_recommendations('The Killer')


# In[2]:


get_recommendations('Jumanji')


# In[3]:


# on fait par la suite la recommandation selon les "keywords"
d4 = pd.read_csv('C:\\Users\\HP\\Desktop\\3AMIndS\\data mining\\NLP\\keywords.csv')

d2 = d2.merge(d4, on='id')

d2.columns


from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    d2[feature] = d2[feature].apply(literal_eval)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        
        if len(names) > 3:
            names = names[:3]
        return names

    return []

d2['director'] = d2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    d2[feature] = d2[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    d2[feature] = d2[feature].apply(clean_data)
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
d2['soup'] = d2.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
d5 = d2['soup'].head(len(d3))

count_matrix = count.fit_transform(d5)
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
d2 = d2.reset_index()
indices = pd.Series(d2.index, index=d2['title'])
get_recommendations('Assassins', cosine_sim2)


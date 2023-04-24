import pandas as pd

import ast

import time

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

movies = pd.read_csv('5000_movies.csv')
creditss = pd.read_csv('5000_credits.csv')

cv = CountVectorizer(max_features=5000, stop_words='english')

movies = movies.merge(creditss,on='title')
li_title = list(movies['title'])


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:            
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
    

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['tags'] = movies['genres'] + movies['cast'] + movies['crew']
movies['genres'] = movies['genres'].apply(lambda r: [i.replace(" ", "") for i in r])
movies['keywords'] = movies['keywords'].apply(lambda r: [i.replace(" ", "") for i in r])
movies['cast'] = movies['cast'].apply(lambda r: [i.replace(" ", "") for i in r])
movies['crew'] = movies['crew'].apply(lambda r: [i.replace(" ", "") for i in r])

df = movies[['movie_id', 'title', 'tags']]

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

print('Hi I am a Recommender System')
print('I can give you recommendations for any Hollywold movie.')
print('Ensure that you enter the correct spelling of the movie')
time.sleep(4)
movie_input = input('Enter the name of the movie for which you want Recommandations:- ').title()
print()
print(f'Here are few Recommendation for the movie:- {movie_input} ')
time.sleep(2)
print()

def recommend(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances = similarity[movie_index] 
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda m:m[1])[1:6]

    for i in movie_list:
        print(df.iloc[i[0]].title)

recommend(movie_input)

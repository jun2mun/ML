# 1. Demographic Filtering (인구 통계학적 필터링) # --> 사용자 신상(연령,성별) 등을 고려한 필터링
# Content Based Filtering (컨텐츠 기반 필터링) # --> 사용자가 클릭한 영화와 연관된 필터링
# Collaborative Filtering (협업 필터링) # 비슷한 취향을 가진 사람들끼리 연관한 필터링


# 1. Demographic Filtering (인구통계학적 필터링) #
import enum
import pandas as pd
import numpy as np
df1 = pd.read_csv('./나도코딩/프로젝트/영화 추천 시스템/tmdb_5000_credits.csv')
df2 = pd.read_csv('./나도코딩/프로젝트/영화 추천 시스템/tmdb_5000_movies.csv')

df1.columns = ['id', 'title', 'cast', 'crew']
df1[['id', 'cast', 'crew']]

df2 = df2.merge(df1[['id', 'cast', 'crew']],on='id')
df2.head(3)

print(df2.loc[0,'genres'])

s1 = [{"id": 28, "name": "Action"}]
s2 = '[{"id": 28, "name": "Action"}]'

from ast import literal_eval
s2 = literal_eval(s2)
s2, type(s2) 

features = ['cast','crew','keywords','genres']
for feature in features:
    #list로 변환하여 넣어주기
    df2[feature] = df2[feature].apply(literal_eval)

df2.loc[0,'crew']

# 감독 정보를 추출
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan 

df2['director'] = df2['crew'].apply(get_director)
# isnull()
df2[df2['director'].isnull()]

# 처음 3개의 데이터 중에서 name 에 해당하는 value 만 추출
def get_list(x):
    if isinstance(x,list):
        names =[i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return [] # 예상하지 못한 데이터 타입이나 수이면

features = ['cast','keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

df2[['title','cast','director','keywords','genres']].head(3)

def clean_data(x):
    if isinstance(x,list): # x가 list인 경우
        # 빈칸이 있으면 빈칸이 없는 데이터로 바꿔서 소문자로 return
        return [str.lower(i.replace(' ','')) for i in x]
    else: # list 아니라 str이면
        if isinstance(x,str):
            return str.lower(x.replace(' ',''))
        else:
            return ''

features = ['cast','keywords','genres']
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

# [i,am,human] => i am human
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' '\
        + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup,axis=1)
df2['soup']

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
count_matrix

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix,count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

#get_recommendations('The Dark Knight Rises',cosine_sim2)
#get_recommendations('Up',cosine_sim2)

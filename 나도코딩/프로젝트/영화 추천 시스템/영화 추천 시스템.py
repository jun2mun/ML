# 1. Demographic Filtering (인구 통계학적 필터링) # --> 사용자 신상(연령,성별) 등을 고려한 필터링
# Content Based Filtering (컨텐츠 기반 필터링) # --> 사용자가 클릭한 영화와 연관된 필터링
# Collaborative Filtering (협업 필터링) # 비슷한 취향을 가진 사람들끼리 연관한 필터링


# 1. Demographic Filtering (인구통계학적 필터링) #
import enum
import pandas as pd
import numpy as np

df1 = pd.read_csv('./나도코딩/프로젝트/영화 추천 시스템/tmdb_5000_credits.csv')
df2 = pd.read_csv('./나도코딩/프로젝트/영화 추천 시스템/tmdb_5000_movies.csv')

#print(df1.head())
# print(df1.shape,df2.shape)
# (4803,4) (4803,20)

# 데이터가 같은지 확인
#df1['title'].equals(df2['title'])
# 같은 데이터가 많은 columns으로ㅗ 병합

print(df1.columns)
# ['movie_id', 'title', 'cast', 'crew']
df1.columns = ['id', 'title', 'cast', 'crew']
df1[['id', 'cast', 'crew']]

# 'id' 기준으로 병합한다. ->
# id 기준으로 나머지 특성들을 column 추가하여 붙인다. 
df2 = df2.merge(df1[['id', 'cast', 'crew']],on='id')
df2.head(3)

# 영화 1: 영화의 평점이 10/10 -> 5명이 평가
# 영화 2: 영화의 평점이 8 /10 -> 500명이 평가

C = df2['vote_average'].mean() # vote_average 평균
C

m = df2['vote_average'].quantile(0.9) # 하위 90프로(=> 상위 10프로) 데이터를 뽑아온다.
m # 1838.....4.4.4.4 

q_movies = df2.copy().loc[df2['vote_average'] >= m] # 1838보다 큰 movie만 복사
q_movies.shape # (481,22) shape
q_movies['vote_count'].sort_values()

def weighted_rating(x,m=m,C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) + (m/ (m+v) * C)

# 새로운 score column 생성
# axis =0 기본값  -> axis =1 => row 단위로 
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score',ascending=False) # score기준으로 내림차순
print(q_movies[['title','vote_count','vote_average','score']].head(10))




### 데이터 시각화 ###
pop = df2.sort_values('popularity',ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6),
 align='center', color='skyblue')

plt.gca().invert_yaxis()
plt.xlabel('Popularity')
plt.title('Popular Movies')
plt.show()



### Content Based Fiiltering (컨텐츠 기반 필터링) ###
df2['overview'].head(5)

## Bag Of Words - BOW ##
# I am a boy (문장 1)
# I am a girl (문장 2)
# I(2), am(2), a(2), boy(1), girl(1)

#           I am a body girl
# 문장 1    1  1  1   1  0   (1,1,1,1,0)
# 문장 2    1  1  1   0  1   (1,1,1,0,1)

# 피쳐 벡터화.

# 문서 100개
# 모든 문서에서 나온 단어 10,000개
# 100 * 10,000 = 100만

"""
    단어1 단어2 단어3 .....
문서1  1    1    0 ..........
문서2
...
...

"""

# 1. TfidVectorizer (TF-IDF 기반의 벡터화) -> 조사 같은거 많은 문장에서 사용됨
# 2. CountVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#print(ENGLISH_STOP_WORDS) # 조사 같은 stop 단어 나옴 -> 이 단어들 무시를 한다.

# null 에 해당하는 단어가 있으면 true
df2['overview'].isnull().values.any()
# Null 값을 자동으로 빈값으로 채워줌
df2['overview'] = df2['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape # (4803,20978) 20978개의 단어로 이루어짐

# cosine 함수보다 linear_kernel 함수가 더 빠름(공식문서)
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix) # x,y값으로 두번 넣음
# cosine_sim
"""
    문장1 문장2 문장3
문장1  1   0.3  0.8
문장2  0.3  1   0.5
문장3  0.8  0.5  1
 문장3과1이 유사도가 가장 큼
"""
# cosine_sim.shape (4803,4803)

# 중복 title 제거 + title index를 가져옴
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
#indices['Avatar'] avatar title이 3번 나옴
#indices.iloc[0]

# 영화의 제목을 입력받으면 코사인 유사도를 통해서
# 가장 유사도가 높은 상위 10개의 영화 목록 반환
def get_recommendations(title,cosine_sim=cosine_sim):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = indices[title]
    # 코사인 유사도 매트릭스 
    # {cosine_sim}에서 idx에 해당하는 데이터를
    # {idx,유사도} 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_socres = sim_scores[1:11]    

    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices = [i[0] for i in sim_scores]

    # 인덱스 정보를 통해 영화 제목 추출
    return df2['title'].iloc[movie_indices]

get_recommendations('Avengers: Age of Ultron')


##############################################3
# 배우 , 감독 등 기반 필터링 #

### 다양한 요소 기반 추천(장르, 감독, 키워드 등) ###

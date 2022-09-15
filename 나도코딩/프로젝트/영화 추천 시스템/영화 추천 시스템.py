# 1. Demographic Filtering (인구 통계학적 필터링) # --> 사용자 신상(연령,성별) 등을 고려한 필터링
# Content Based Filtering (컨텐츠 기반 필터링) # --> 사용자가 클릭한 영화와 연관된 필터링
# Collaborative Filtering (협업 필터링) # 비슷한 취향을 가진 사람들끼리 연관한 필터링


# 1. Demographic Filtering (인구통계학적 필터링) #
import pandas as pd
import numpy as np

df1 = pd.read_csv('./나도코딩/프로젝트/영화 추천 시스템/tmdb_5000_credits.csv')
df2 = pd.read_csv('./나도코딩/프로젝트/영화 추천 시스템/tmdb_5000_movies.csv')

print(df1.head())
# 클러스터링 : '유사한 특성을 가지는 애들끼리 군집화'

# K-평균 
# 동작 순서
# 1. k값 설정
# 지정된 k개 만큼의 랜덤 좌표 설정
# <아래 반복>
# 모든 데이터로부터 가장 가까운 중심점 선택
# 데이터들의 평균 중심으로 중심점 이동
# 중심점이 더 이상 이동되지 않을 때까지 반복

# K-means는 초기 centroid 위치에 결과값이 영향을 많이 받음

# 대안책
# K-menas++
# 데이터 중에서 랜덤으로 1개를 중심점으로 선택
# 나머지 데이터로부터 중심점까지의 거리 계산
# 중심점과 가장 먼 지점의 데이터를 다음 중심점으로 선택
# 중심점이 K 개가 될때까지 반복

# K-means 전통적인 방식으로 진행

# Optimal K

## Elbow Method ##
# 1. k 변화에 따른 중심점까지의 평균 거리 비교
# 2. 경사가 완만해지는 지점의 k 선정

## Euclidean Distance ## (유사도 측정시)
## Manhattan Distance ## 
## Cosine Similarity ##

##################################################################

### 경고 발생시 ###
import os
os.environ['OMP_NUM_THREADS'] = '1'

# K-means
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./KMeansData.csv ')

# 비지도 학습이여서 Y가 필요없다.
X = dataset.iloc[:,:].values
# X = dataset.values도 똑같음 //////  dataset.to_numpy() # 공식홈페이지 권장
X[:5]

## 데이터 시각화 (전체 데이터 분포 확인) ### 단위가 다르기때문에 시각화 단위수정 필요!! ###
plt.scatter(X[:,0],X[:,1]) # x축 : hour , y축 : score
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

## 데이터 시각화 (축 범위 통일)
plt.scatter(X[:,0],X[:,1]) # x축 : hour , y축 : score
plt.title('Score by hours')
plt.xlabel('hours')
plt.xlim(0,100)
plt.ylabel('score')
plt.ylim(0,100)
plt.show()


## 피처 스케일링 (Feature scaling) ##
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X[:5]

## 데이터 시각화 (스케일링된 데이터) ##
plt.figure(figsize=(5,5))
plt.scatter(X[:,0],X[:,1]) # x축 : hour , y축 : score
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

################################################################

# 최적의 K 값 찾기 (엘보우 방식 Elbow Method)
from sklearn.cluster import KMeans
inertia_list = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0) # init 설정 안하면 일반 K-mean이여서 안좋을 수도 있음
    kmeans.fit(X) # 비지도 학습이므로 Y 필요없음
    inertia_list.append(kmeans.inertia_) # 각 지점으로부터 클러스터의 중심(centroid) 까지의 거리의 제곱의 합

plt.plot(range(1,11),inertia_list)
plt.title('Elbow Method') # 기울기가 확떨어지다가 완만해지는 시점ㅇ을 K로 설정하자
plt.xlabel('n_clusters')
plt.ylabel('inertia')
plt.show()

K = 4 # 최적의 K값

kmeans = KMeans(n_clusters=K, random_state=0)
#kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X) # X데이터를 넣어 fit 한후, 예측되는 값을 반환
y_kmeans

## .데이터 시각화 (최적의 K) ##
centers = kmeans.cluster_centers_  # 클러스터의 중심점 (centroid) 좌표
for cluster in range(K):
    plt.scatter(X[y_kmeans == cluster,0],X[y_kmeans == cluster, 1],s=100,edgecolors='black') # X중에서 y_kmeans와 동일한 애들 추출 
    plt.scatter(centers[cluster,0], centers[cluster,1], s=300, edgecolors='black', color='yellow', marker='s')
    plt.text(centers[cluster,0],centers[cluster,1],cluster,va='center',ha='center') # 클러스터 텍스트 출력

plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

## 데이터 시각화 (스케일링 원복 ) ##
X_org = sc.inverse_transform(X) # Feature Scaling 된 데이터를 다시 원복
X_org[:5]

centers_org = sc.inverse_transform(centers)
centers_org

centers = kmeans.cluster_centers_  # 클러스터의 중심점 (centroid) 좌표
for cluster in range(K):
    plt.scatter(X[y_kmeans == cluster,0],X[y_kmeans == cluster, 1],s=100,edgecolors='black') # X중에서 y_kmeans와 동일한 애들 추출 
    plt.scatter(centers[cluster,0], centers[cluster,1], s=300, edgecolors='black', color='yellow', marker='s')
    plt.text(centers[cluster,0],centers[cluster,1],cluster,va='center',ha='center') # 클러스터 텍스트 출력

plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
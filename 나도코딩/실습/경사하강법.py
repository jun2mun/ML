# 흐름 이해 #

# 일반적으로 사용하는 학습률 (0.001,0.003, 0.01,0.03, 0.1,0.3)
# 학습률이 커질수록 loss 가 
# Epoch 
# 경사하강법 ( 전체 데이터셋에 하기에는 연산이 늘어남) =-> stochastic (확률적) 경사하강법 사용

import sklearn
import os
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./LinearRegressionData.csv ')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

## 경사 하강법 ##
from sklearn.linear_model import SGDRegressor # SGD : Stochastic Gradient Descent 확률적 경사 하강법
# 지수 표기법
# 1e-3 : 0.001 (10^-3)
# 1e-4 : 0.0001 (10^-4)
# 1e+3 : 1000 (10^3)
# 1e+4 : 10000 (10^4)

#sr = SGDRegressor() # default임


### max_iter : 훈련 세트 반복 횟수 (Epoch 횟수)
### eta0 : 학습률 (learning_rate)
'''
max_iter eta0 조절을 통해 
'''
sr = SGDRegressor(max_iter=1000, eta0=0.001, random_state=0, verbose=1)# verbose 1하면 결과가 보임?

sr.fit(X_train,y_train)

'''
plt.scatter(X_train,y_train,color='blue') # 산점도
plt.plot(X_train,sr.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours (train data, SGD)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') #  Y 축 이름
plt.show()
'''

#print(sr.coef_, sr.intercept_) # 기울기 , y절편

sr.score(X_test, y_test) # 테스트 세트를 통한 모델 평가

sr.score(X_train, y_train) # 훈련 세트를 통한 모델 평가

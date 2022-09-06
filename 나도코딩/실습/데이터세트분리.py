import sklearn
import os
#Linear Regression
# 공부 시간에 따른 시험 점수
#print(os.getcwd())
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./LinearRegressionData.csv ')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
# 독립변수, 의존 변수 넣기, 훈련 80 : 테스트 20으로 분리 

print(X, len(X))

print(X_train, len(X_train)) # 훈련 세트 X, 개수

print(y, len(y)) # 전체 데이터 y

print(y_train, len(y_train)) # 훈련 세트 y

print(y_test, len(y_test)) # 테스트 세트 y

## 분리된 데이터를 통한 모델링 ##

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train) # train set 애들만 훈련

## 데이터 시각화(훈련 세트) ##
'''
plt.scatter(X_train,y_train,color='blue') # 산점도
plt.plot(X_train,reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') #  Y 축 이름
plt.show()
'''
## 데이터 시각화 (테스트 세트 ) ##
'''
plt.scatter(X_test,y_test,color='blue') # 산점도
plt.plot(X_train,reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') #  Y 축 이름
plt.show()
'''

#print(reg.coef_)
#print(reg.intercept_)


## 모델 평가 ##
#reg.score(X_test, y_test) # 테스트 세트를 통한 모델 평가
#reg.score(X_train,y_train) # 훈련 세트를 통한 모델 평가


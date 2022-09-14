# 다항 회귀 #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./PolynomialRegressionData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

## 3-1. 단순 선형 회귀 (Simple Linear Regression)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y) # 전체 데이터로 학습

# 데이터 시각화(전체)

plt.scatter(X,y,color='blue') # 산점도
plt.plot(X,reg.predict(X),color='green') # 선 그래프
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

#print(reg.score(X,y)) # 전체 데이터를 통한 모델 평가


## 3-2. 다항 회귀 (Polynomial Regression)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) # 2차 다항식 --> 4차로 바꿔보기도 하자
X_poly = poly_reg.fit_transform(X)
## fit_transform함수는 아래 fit + transform 함수 를 한번에 실행하는 것
#poly_reg.fit()
#poly_reg.transform(X)

X_poly[:5] # [x] -> [x^0, x^1, x^2] -> x 가 3이라면 [1, 3, 9] 으로 변환

poly_reg.get_feature_names_out()


lin_reg = LinearRegression()
lin_reg.fit(X_poly,y) # 변환된 X 와 y 를 가지고 모델 생성 (학습)

### 데이터 시각화 (변환된 X 와 y)
plt.scatter(X,y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)),color='green')
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

X_range = np.arange(min(X),max(X),0.1) # 최소 값 - 최대값 까지의 범위를 0.1 단위로 잘라서 데이터를 생성
X_range.shape() # (46,)
X_range.reshape(-1, 1) # -1 + column 개수를 넣어주면 -> row 개수는 자동으로 계산 , column 개수는 1개 // (46,1)
X_range[:5]

plt.scatter(X,y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X_range)),color='green')
plt.title('Score by hours (genius)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

# 공부 시간에 따른 시험 성적 예측
reg.predict([[2]]) # 2시간을 공부했을 때 선형 회귀 모델의 예측
lin_reg.predict(poly_reg.fit_transform([[2]])) # 2시간을 공부했을 때 다항 회귀 모델의 예측


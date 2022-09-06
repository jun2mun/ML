import sklearn
import os
#Linear Regression
# 공부 시간에 따른 시험 점수
#print(os.getcwd())
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./LinearRegressionData.csv ')
dataset.head()

X = dataset.iloc[:,:-1].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수)
y = dataset.iloc[:,-1].values  # 마지막 칼럼 데이터 (종속 변수 - 결과)

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X,y) # 학습 (모델 생성)

y_pred = reg.predict(X) # X에 대한 예측 값

'''
plt.scatter(X,y,color='blue') # 산점도
plt.plot(X, y_pred, color='green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') #  Y 축 이름
plt.show()
'''
#print('9시간 공부했을 때 예상 점수 : ', reg.predict([[9]]))  
# # X가 2차원 배열 이니까 이차원 배열로 넣어줌 [[9],[8]]

#print(reg.coef_) # 기울기 (m)
#print(reg.intercept_) # y 절편 (b)

#y = mx + b -> y = reg.coef_ * x - reg.intercept_


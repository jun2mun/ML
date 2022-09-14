# 시그모이드 함수가 로지스틱 회귀에서 사용되는 이유를 이해하자. #

# 4. Logistic Regression #
# 공부 시간에 따른 자격증 시험 합격 가능성

import numpy as np
import matplotlib.pyplot as ply
import pandas as pd

dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./LogisticRegressionData.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 


### 데이터 분리 ###
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# 학습(로지스틕 회귀 모델)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# 6시간 공부했을 때 예측?
classifier.predict([[6]])
# 결과 1 : 합격할 것으로 
classifier.predict_proba([[6]])
# 합격할 활률 출력
# 불합격 확률 ()% , 합격 활률 ()%

# 4시간 공부했을 때 예측?
classifier.predict([[4]])
# 결과 0 : 불합격할 것으로 예측

## 분류 결과 예측 (테스트 세트)
y_pred = classifier.predict(X_test)
y_pred # 예측 값

y_test # 실제 값 (테스트 세트)
X_test # 공부 시간 (테스트 세트)

classifier.score(X_test,y_test) # 모델 평가 (회귀 모델과 다르게 분류 모델은 평가가 딱 떨어짐)
# 전체 테스트 세트 4개 중에서 분류 예측을 올바로 맞힌 개수 3/4 = 0.75 ()

### 데이터 시각화 (훈련 세트) ###
X_range = np.arange(min(X),max(X), 0.1) # X의 최소값에서 최대값까지를 0.1 단위로 잘라서 데이터 생성
X_range

# p = 1 / (1 + np.exp(-y)))  || # y = mx + b
p = 1 / (1 + np.exp(-(classifier.coef_ * X_range + classifier.intercept_)))
print(p)
p.shape
X_range
p = p.reshape(-1) # 1 차원 배열 형태로 변경
p.shape

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range),0.5), color='red') # X_range 개수만큼 0.5로 가득찬 배열 만들기
plt.title('Probability by hours')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()


## 데이터 시각화 (테스트 세트) ##
plt.scatter(X_test,y_test, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range),0.5), color='red') # X_range 개수만큼 0.5로 가득찬 배열 만들기
plt.title('Probability by hours')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()


classifier.predict_proba([[4.5]]) # 4.5시간 공부했을 때 확률 (모델에서는 51% 합격 예측, 실제로는 불합격)

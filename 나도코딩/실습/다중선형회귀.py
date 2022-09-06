## 다중 선형 회귀란 ##

## simple Linear Regression (단순 선형 회귀) y = mx + b
## Multiple Linear Regression (다중 선형 회귀) y = b + m1 x1 + m2 x2 ....


## 원 핫 인코딩

# 예) 공부장소(Home,Library,cafe) => (1,0,0)과 같이 표시
# 공부장소 column 옆에 dummy 칼럼 home,library,cafe를 생성하여 해당하는 장소에 해당하면 1로 표시


## 다중 공선성(Multicollinearity)
"""
독립 변수들 간에 서로 강한 상관관계를 가지면서
회귀계수 추정의 오류가 나타나는 문제  ====>>>>   하나의 피처가 다른 피처에 영향을 미침
"""
# Dummy Column 이 n개면? n-1개만 사용


############################################################################################
## 원핫 인코딩 ##
import pandas as pd
dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./MultipleLinearRegressionData.csv ')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),[2])], remainder='passthrough')
# drop first를 사용함으로써 다중 공선성 없앰
# [2]번 칼럼에다가만 적용
# remainder => 원핫 인코딩을 적용하지 않는 칼럼들을 'passthrough'(그대로 둔다) 한다. 
X = ct.fit_transform(X) # X 원핫 인코딩 적용
#print(X)

# 1 0 : Home
# 0 1 : Library
# 0 0 : Cafe
#############################################################################################
## 데이터 세트 분리 ##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
##  학습 다중 선형 회귀 ##
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# 예측 값과 실제 값 비교 (테스트 세트) #
y_pred = reg.predict(X_test)
#print(y_pred)
#print(reg_coef_,reg_intercept_)


# 모델 평가
reg.score(X_train, y_train) # 훈련 세트
reg.score(X_test,y_test) # 테스트 세트
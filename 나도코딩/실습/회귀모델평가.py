# Evaluation
# MAE(Mean Absolute Error): 실제 값과 예측 값 차이의 절대값들의 평균 
# MSE !!! (Mean Squared Error) : 실제 값과 예측 값 차이의 제곱한 값들의 평균
# RMSE (Root Mean Squared Error) : MSE에 루트 적용


# total = reg + error // Stotal = Sreg + Serror // SStotal = SSreg + SSerror
# error : (실제 값 - 예측 값) Serror error^2  SSerror sigma(Serror)
# reg : (예측 값 - 평균 값) Sreg seg^2  SSreg sigma(Sreg)
# total : (실제 값 - 평균 값) Stotal total^2  SStotal sigma(Stotal)

# R Square : 결정계수 (데이터의 분산을 기반으로 한 평가 지표)
# R^2 = 1 - SSE/SST = SSR/SST
# 1에 가까울수록 좋다. // 0에 가까울수록 나쁘다.

######################################################################################
                                # 테스트 용 ##
## 원핫 인코딩 ##
from statistics import mean
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
###################################################################################
# 평가지표
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_pred)) # 실제 값, 예측 값 # MAE

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred, squared=False)) # RMSE

from sklearn.metrics import r2_score
r2_score(y_test,y_pred) 
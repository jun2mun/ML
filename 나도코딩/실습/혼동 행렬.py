
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


## 분류 결과 예측 (테스트 세트)
y_pred = classifier.predict(X_test)
y_pred # 예측 값


# 혼돈 행렬 (Confusion Matrix) #

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# TRUE NEGATIVE(TN)     FALSE POSITIVE(FP)
# 불합격일거야 (예측)    합격일거야 (예측)
# 불합격 (실제)          불합격(실제

# FALSE NEGATIVE (FN)    TRUE POSTIVE(TP)
# 불합격일거야 (예측)     합격일거야(예측)
# 합격 (실제)             합ㄱ격(실제)

# array([[1,1],
#         [0,2], dtype=int64])

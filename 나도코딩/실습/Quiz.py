import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1)
dataset = pd.read_csv('./나도코딩/실습./PythonMLWorkspace(LightWeight)./ScikitLearn./QuizData.csv ')
total = dataset.iloc[:,:-1].values # Total 출력 || X
reception = dataset.iloc[:,-1].values # reception 출력 || y

# 2)
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(total,reception,test_size=0.25,random_state=0)

# 3)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)


# 4 train set)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='green')
plt.title('wedding reception (train)')
plt.xlabel('total')
plt.ylabel('reception')
plt.show()

# 5 test set) ## 복습 ##
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_train,reg.predict(X_train),color='green')
plt.plot(X_test,reg.predict(X_test),color='red')

plt.title('wedding reception (test)')
plt.xlabel('total')
plt.ylabel('reception')
plt.show()

# 6 모델 평가 점수

# 훈련 세트 평가 점수
#print(reg.score(X_train,y_train))

# 테스트 세트 평가 점수
#print(reg.score(X_test,y_test))

total = 300
y_pred = reg.predict([[total]])
print(f'결혼식 참석 인원 {total} 명에 대한 예상 식수 인원은 {np.around(y_pred).astype(int)} 명입니다.') # np.around 반올림

#
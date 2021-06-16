# R2 : 결정 계수

# MLP 란?

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape)
print(y.shape)

x = np.transpose(x)
print(x)
print(x.shape)

# 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(2,))) 
model.add(Dense(30))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x, y, batch_size=1, epochs=700)

# 예상 및 결과
result = model.evaluate(x, y, batch_size=1)
print("result : ", result)

# test = np.array([[11, 12, 13], [21, 22, 23]])
# test = np.transpose(test)

# predict = model.predict(test)
# print("predict : ", predict)

y_predict = model.predict(x) # x 에 대한 예측은 x 의 결과값이 나올것

# 결과에서 e-0n 은 앞에 n 개 만큼 0이 있다는 뜻이다
# RMSE 는 그대로 'rmse' 라고 지정해줄 수는 없고
# sklearn 에서 제공하는 제곱함수를 통해서 제곱시킨뒤 사용해야 함

from sklearn.metrics import mean_squared_error # mse 를 의미함

# y_test 는 새 결과 값, y_predict 는 evaluate 로 나온 예측값
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y, y_predict))
print("mse : ", mean_squared_error(y, y_predict))

# y_predict 는 모델에 대한 예측 결과값을 의미한다

from sklearn.metrics import r2_score

# y_test : y 의 원래 값, y_predict : 예측 값
print('R2 : ', r2_score(y, y_predict)) # r2 가 높은건 그은 직선에 거의 인접해 있는 점들이 많다는 소리임
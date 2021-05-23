import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

# 과제0. github 만들기 kera 레파지토리!!
# 과제1. 네이밍룰 알아오기

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # 배열이 하나의 차원으로 되어있으니까 dim=1
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(1))
# 히든 layer는 건드려도 되지만 input,output은 건드리면 안된다

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) # 데이터수가 배치사이즈보다 작으면 한번에 다 들어간다.

#4. 평가, 예측
loss = model.evaluate(x_test,y_test, batch_size=1)
print('loss :', loss)

results = model.predict([9]) # []의 이유: predict도 1차원이어야 하니까
print('result :', results)
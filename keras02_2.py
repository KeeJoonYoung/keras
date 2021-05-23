import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])
# train과 test의 w,b가 다름

x_predict = np.array([111,112,113])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
# 딥하지 않은 딥러닝

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1) # 배치사이즈가 더 커도 됨

#4. 평가,예측
loss= model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

results = model.predict(x_predict)
print('results : ',results)

# 과제3 잘 만들어라
# train으로 수럼되는지 test로 수렴되는지를 알고 싶은거임
# keras26 커피
import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# x_pred = np.array([50,60,70])

# 실습, 코딩하시오 LSTM
# 내가 원하는 답은 80이다

print(x.shape) # (13,3)
print(y.shape) # (13,)

x = x.reshape(13, 3, 1) 
print(x.shape) # (13, 3, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(3, 1), return_sequences=True))  
model.add(LSTM(10))
# LSTM을 두 단 쓰려면 위에 return_sequences를 넣으면 된다
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# model.summary()
# 결과가 더 좋아질수도 있고 아닐수도 있다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([50,60,70]) #(3,)인데 이걸 (1,3,1)로 바꾸어야 한다. [[[5],[6],[7]]] 이 된다.
x_pred = x_pred.reshape(1,3,1)


y_pred = model.predict(x_pred)
print(y_pred)

############################################## 김동훈. 70 후반이면 잘 나오는거다

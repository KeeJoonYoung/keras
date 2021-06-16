# 25_LSTM에서 커피
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])  # LSTM으로 만들고 싶은면 reshape해서 (4,3,1)만들어준다 그리고 행무시니까 16line에 (3,1)이 input_shape가 된다
y = np.array([4,5,6,7]) 
print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1) # (4, 3, 1)
print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(3, 1)))  # 여기서 3,1은 (4,3)의 3, 1은 1개씩 연산 (몇개씩 잘라서 연산을 하는지)
# # dense는 데이터가 2차원(N,열), image는 4차원(N,가로,세로,칼라) 지금 하는 시계열=LSTM은 3차원(N, , )
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([5,6,7]) #(3,)인데 이걸 (1,3,1)로 바꾸어야 한다. [[[5],[6],[7]]]이 된다.
x_pred = x_pred.reshape(1,3,1)

################################ 지금까지 배운거를 토대로 답을 8의 근사치로 만들어라

y_pred = model.predict(x_pred)
print(y_pred)

'''
이건 회귀이므로 숫자로 나올 것이다. 잘하면 엄청 잘 하는데 잘 안나올 수 도? 결과가 8의 근사치가 나와야함 7.999까진 해라
'''
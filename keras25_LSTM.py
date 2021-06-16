# RNN 대강 공부해오기, LSTM도
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # (4, 3) # LSTM으로 만들고 싶은면 reshape해서 (4,3,1)만들어준다 그리고 행무시니까 (3,1)이 input_shape가 된다
y = np.array([4,5,6,7]) # (4,)
print(x.shape)
print(y.shape)

x = x.reshape(4, 3, 1) # (4, 3, 1)
print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(3, 1)))  # 여기서 3,1은 (4,3)의 3, 1은 1개씩 연산 (몇개씩 잘라서 연산을 하는지)
# # dense는 데이터가 2차원(N,열), image는 4차원(N,가로,세로,칼라) 지금 하는 시계열=LSTM은 3차원(N, , )
model.add(Dense(10))
model.add(Dense(1))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480          480?
_________________________________________________________________
dense (Dense)                (None, 10)                110          110?
_________________________________________________________________  
dense_1 (Dense)              (None, 1)                 11           11?
=================================================================
Total params: 601
Trainable params: 601
Non-trainable params: 0
_________________________________________________________________
'''
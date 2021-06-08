# 이진분류

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)   569행 30열

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)
# print(x[:5]) # 한개당 30개의 coluom이 있다.
# print(y)
# print(np.max(y), np.min(y)) # 1 0

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
############################## 튜닝 해야함
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(1, activation='sigmoid')) # 시그모이드는 layer을 통과하는 모든 값을 0과 1사이로 수렴시킨다.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # binary_crossentropy 0이나 1로 바꿈(0.5기준 반올림)
model.fit(x,y,epochs=50, validation_split=0.2)

results = model.evaluate(x,y)
print('loss : ', results[0])
print('metrics : ',results[1])

y_pred = model.predict(x[-5:-1]) # 맨뒤에서부터 -1이 가장 뒤에, 그로부터 순서. 가장 끝의 5개의 데이터
print(y_pred)
print(y[-5:-1])

'''
loss :  63.17013168334961
metrics :  63.17013168334961
[[-12.073573 ]
 [-11.232173 ]
 [ -3.9850032]
 [ -5.9003553]]
[0 0 0 0] - 처음 상태binary_crossentropy 안한거
'''

'''
loss :  0.6457214951515198
metrics :  0.9138839840888977
[[1.4335650e-38]
 [2.4887246e-29]            다 0.5미만이므로 0으로 판단
 [1.0489806e-08]
 [1.1538378e-29]]
[0 0 0 0]
'''
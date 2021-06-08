# 이진분류 -> 다중분류로 수정하기
# 튜닝 필요
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

## 원핫인코딩 OneHotEncoding
# from tensorflow.keras.utils import to_categorical

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(20)) 
model.add(Dense(20)) 
model.add(Dense(20)) 
#model.add(Dense(1, activation='sigmoid')) # 시그모이드는 layer을 통과하는 모든 값을 0과 1사이로 수렴시킨다.
model.add(Dense(1, activation='softmax')) # 라벨의 개수가 마지막 노드의 개수가 된다.

#3. 컴파일, 훈련
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # binary_crossentropy 0이나 1로 바꿈(0.5기준 반올림)
##########################33                          확인 요망
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  #29라인에 마지막 노드가 2이면 안되고 1이어야함
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) #29라인에 마지막 노드가 2이어야 함 1은 결과가 이상함
model.fit(x, y, epochs=100, validation_split=0.2)


results = model.evaluate(x,y)
print('loss : ', results[0])
print('accuracy : ',results[1])

y_pred = model.predict(x[-5:-1]) # 맨뒤에서부터 -1이 가장 뒤에, 그로부터 순서. 가장 끝의 5개의 데이터
print(y_pred)
print(y[-5:-1])

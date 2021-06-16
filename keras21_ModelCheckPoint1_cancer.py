# 가장 좋은 점을 check하겠다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.callbacks import ModelCheckpoint

#1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)   569행 30열

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

## 원핫인코딩 OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# y= to_categorical(y)   # (150, ) ->(150,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=66
)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(20)) 
model.add(Dense(20)) 
model.add(Dense(20)) 
#model.add(Dense(1, activation='sigmoid')) # 시그모이드는 layer을 통과하는 모든 값을 0과 1사이로 수렴시킨다.
model.add(Dense(2, activation='softmax')) # 라벨의 개수가 마지막 노드의 개수가 된다.

#3. 컴파일, 훈련
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # binary_crossentropy 0이나 1로 바꿈(0.5기준 반올림)
##########################33                          확인 요망
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  #29라인에 마지막 노드가 2이면 안되고 1이어야함
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) #29라인에 마지막 노드가 2이어야 함 1은 결과가 이상함 loss가 none, 다른거 하나가 0이 나온대

# early stopping(훈련을 정지시키는 거다.)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto') # 갱신이 10번까지 되지 않으면(10번 참겠다) mode가 max일때는 accuracy이다. 근데 보통 loss가 더 좋다. max인지 min인지 모르겠으면 auto로 하면 된다
# loss 보다 val_lose로 하는게 좋긴함
modelpath = './keras/CheckPoint/k21_cancer_{epoch:02d}-{val_loss:.4f}.hdf5' # 두자리로 epochs가 들어가고 소수 4자리수로 val_loss가 들어온다
cp= ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                     save_best_only=True, mode='auto') # 여기선 min이 된다



hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.2, # validation: 20프로의 데이터를 평가하겠다
            callbacks=[early_stopping, cp]) # 새로 만든 cp는 callbacks에 들어간다

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
print('accuracy : ',results[1])

y_pred = model.predict(x_test[-5:-1]) # 맨뒤에서부터 -1이 가장 뒤에, 그로부터 순서. 가장 끝의 5개의 데이터
print(y_pred)
print(y_test[-5:-1])

print(hist) # <tensorflow.python.keras.callbacks.History 가 결과로 나온다.
print(hist.history.keys()) # dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])  이거는 훈련 할때 나오던 것들. 
# print(hist.h*istory['loss'])  # [3.550455331802368, 0.8527161478996277, 0.5708760023117065, 0.4006132483482361, 0.26738396286964417]
print(hist.history['val_loss'])  # [1.5223042964935303, 0.2871544361114502, 0.825431764125824, 0.16834567487239838, 0.15253503620624542]


# 시각화 / 그래프
import matplotlib.pyplot as plt
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['train acc', 'val_acc'])
plt.show()

'''
결과는 왼쪽에서 CheckPoint에 새로 만들어진 파일들을 ㅇ보면서 그래프와 비교해서 확인하면 된다. 갱신될때 큰거를 저장하는 과정이다
'''
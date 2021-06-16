# 16_2 커피

import numpy as np
from tensorflow.keras.datasets import mnist # mnist: 손 글씨 흑백 데이터 28*28 x.shape= (6만,28,28,1), y.shape= (10, )

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 이미 나누어져있음

print(x_train.shape, x_test.shape) # (60000, 28, 28): 가로세로 28의 6만개 (10000, 28, 28)  # 28*28*1의 1은 생략, 흑백이라. 컬라라면 뒤에 적어줌
print(y_train.shape, y_test.shape) # (60000,) (10000,)

# minmax scalar: 정규화, 전처리. 
# 0~255를 최대값으로 나누어서 0~1로 바꾸어 아무리 곱해도 1을 넘지 않게 만든다. 라인 14

# x_train = x_train.reshape(60000,28,28,1)
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
# x_test = x_test.reshape(10000,28,28,1)
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.
# x_train = x_train.reshape(60000,14,14,4) # 28*28 하나짜리를 펼쳐서 14*14를 4개로 만든다는 뜻. 전체개수가 달라지면 안된다.
# x_train = x_train.reshape(60000,784) # 이미지로도 dnn도 가능
print(x_train.shape) # (60000, 28, 28, 1)



#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1))) #input_shape=(28,28,1)에서 1은 흑백이니까. 컬라는 3.  # 그다음 layer에 N,28,28,30을 준다
#input_shape와 x_train.shape가 안맞음 그래서 라인13추가
model.add(Conv2D(20, (2,2))) # 두번째부터는 input_shape를 표시 할 필요 없다
model.add(Conv2D(20, (2,2))) # 추가 가능
model.add(Flatten())
model.add(Dense(100, activation ='relu'))
model.add(Dense(10, activation='softmax'))

# model.save('./keras/Model/k23_1_model_1.h5') # fit하기 전에 저장 모델만 저장, 가중치 저장은 안됨
##### 여기서는 저장을 다르게 한다. .이 keras가 된다. 파일이 있는 위치
model.save('./Model/k23_1_model_1.h5')

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # 얘는 그냥 해도 되는데 밑에꺼는 원핫인코딩을 해야함
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# y를 원핫인코딩을 해줘야함
######################################## 마무리 필요
model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)
# 데이터가 크니까 batch_size를 크게 하자 디폴트는 32개

model.save('./keras/Model/k23_1_model_2.h5') # fit하고 난 다음에 저장, model이랑 weight까지 저장

# 두개가 저장된다. 두개의 차이를 알아야한다. load로 아라보자





'''
#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("loss: ", results[0])
print("acc: ", results[1])
'''
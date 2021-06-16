# 23_1 커피

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
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# model = load_model('./keras/Model/k23_1_model_1.h5') # 이거는 에러가 남 # fit하기 전에 저장 한거, 모델만 저장, 가중치 저장은 안됨
# 위에 얘를 사용하려만 컴파일 하고 해야함 밑의 주석 묶음 해제해야함
model = load_model('./keras/Model/k23_1_model_2.h5') # 얘는 잘 나옴 # fit하고 난 다음에 저장 한거, model이랑 weight까지 저장

model.summary()



'''
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # 얘는 그냥 해도 되는데 밑에꺼는 원핫인코딩을 해야함
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)
# 데이터가 크니까 batch_size를 크게 하자 디폴트는 32개

model.save('./keras/Model/k23_1_model_1.h5')
'''


#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("loss: ", results[0])
print("acc: ", results[1])

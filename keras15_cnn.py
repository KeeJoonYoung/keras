from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.python.keras.layers.core import Flatten # 왜 2D? - 그림이 2D이다. 입체는 3D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1, input_shape=(5,5,1))) # (5,5,1)짜리 이미지를 준비 했다. (N,5,5,1) N은 몇장, 1은 컬러
# strides: 자르는거 하나씩 이동하는거.1은 한줄겹치고 2는 겹치지 않게 옆으로 간다.. kernel: 자르는거의 가로세로 크기. filter=10: mode.add(Dense(10,input_shape=(5,))) 10은 아웃풋 똑같이 누적시켜서 나간다
model.add(Conv2D(5, (2,2), padding='same')) 
# 첫번째가 filter라고 인식한다. (2,2): kernel_size. stride는 디폴트가 1이라 안써도 1로 인식함. padding='same': 가의 데이터를 똑같은 횟수로 카운트하게 한다 따라서 input_shape가 (4,4,10)이된다 그다음은 4,4,5

model.add(Flatten()) # Dense에 넣으려고 평평하게 만드는거. 한줄씩 하는거
model.add(Dense(1))  # 다중 분류면 Dense(10,softmax)로 바뀜. 10은 분류의 개수

model.summary()
'''_____________________________________5,5,1을 넣었더니____________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50           5,5,1에서 하니씩 줄어서 4,4이 되고 filter가 10이니까 세번재가 10이 된다. filter가 들어오는거임
                                                                    왜 50인지? 5*10
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 4, 4, 5)           205          same이니까 4,4 그대로 오고 두번째의 필터가 5니까 5. 항상 나가는거에 맞춰서 한다.
                                                                    왜 205인지? (4+1)*41= 5*(40+1)
_________________________________________________________________
flatten (Flatten)            (None, 80)                0            flatten을 하면 4*4*5=80 - 5 by 5 
_________________________________________________________________
dense (Dense)                (None, 1)                 81           최종layer은 항상 Dense이므로 flatten을 해야한다. 만약 1이 2라면 activation을 넣어주어야함.
=================================================================
Total params: 336
Trainable params: 336                                               왜 336인지? 50+205+81 = 336
Non-trainable params: 0
_________________________________________________________________
그림을 받아서 수치화 하는데 그걸 쫙 펼치는거: platten
이미지도 숫자다. 잘라서 구별한다.
결과는 회귀로 나온다.
'''

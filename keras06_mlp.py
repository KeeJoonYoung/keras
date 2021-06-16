# MLP 란?

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape)
print(y.shape)

'''
    위의 데이터는 입력값 x 와 출력값 y 에 대한 차원이 서로 맞지 않다
    x 는 shape 출력시 (2, 10) 으로 나오지만
    y 는 shape 출력시 (10, ) 으로 나온다
    그래서 서로 맞추려면 
    x 를 (10, 2) 로 바꿔야 한다
    y 를 만약 행렬로써 표시한다면 (실제 행렬은 아니지만)
    위에서 아래로 
    1
    2
    3
    4
    ...
    10
    이 될것이다
    근데 지금 x 는 [1, 2, ...., 10], [11, 12, ......, 20] 이 되지만
    이거를 
    [
        [1, 11], 
        [2, 12] 
        ... 
        [10, 20]
    ] 으로 바꿔야 변환이 가능함
    이변환에 쓰이는게 np.transpose() 함수다
'''

x = np.transpose(x)
print(x)
print(x.shape)


# 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(2,))) 
model.add(Dense(30))
model.add(Dense(1))
# input_dim 은 그냥 쉽게 생각하면 x 의 column 갯수를 의미함
# input_shape 도 input_shape = (2, ) 만 쓰면 됨 (그냥 열 갯수만 적어주는 것)
# 근데 또 4차원은 (n, m, k) 이렇게 써야 하는데 쉽게 생각하면 그냥 맨 앞에 꺼만 뺀다 라고 생각하면 됨
# 3차원은 (n, m) 이런식으로 input_shape 를 넣으면 되고
# input_dim 은 2차원 이하의 데이터에서만 가능함 이게 중요
# 그리고 출력 노드는 y 의 차원에 따라 달라짐

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=300)

# [[11, 12, 13], [21, 22, 23]] 값 예측하기
predict = model.predict([[11, 12, 13], [21, 22, 23]], batch_size=1)
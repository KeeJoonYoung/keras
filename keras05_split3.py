# validation split 을 이용한 데이터 분류

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

from sklearn.model_selection import train_test_split

# 데이터 
x = np.array(range(1, 101))
y = array(range(101, 201))

# train_test_split 함수를 한번만 쓰고 validation 을 잡으려면
# fit 함수에 validation_split = 비율 로 지정하면 된다
# train_test_split 의 합이 1을 넘으면 오류남
x_train, x_test, y_train, y_test = train_test_split( 
    x, y, train_size=0.8, test_size=0.2, shuffle=True
)

# 모델
model = Sequential()
model.add(Dense(500, input_dim=1, activation='relu'))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(350))
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')

# train 데이터에서 20%를 validation 으로 쓰겠다는 의미가 된다
model.fit(x_train, y_train, epochs=700, batch_size=1, validation_split=0.2, validation_batch_size=1)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

predict = model.predict([101, 102, 103])
print(predict)


'''
    여지까지는 1차 함수 
    y = wx + b 만 다뤘다
    그러나 같은 1차 함수라도 y = w1x1 + w2x2 + w3x3 가 있을 수 있다
   x1, x2, x3 는 용어로 column 혹은 feature 라 부른다
   '열 우선, 행 무시'
   만약 위에 처럼 y = w1x1 + w2x2 + w3x3 로 구성되어 있다면
   input_dim 이 1 이 아니라 3이 된다
   예를들어 어떤 성적의 데이터들을 갖고 있다면 
   국어 성적
   영어 성적
   수학 성적 등... 여러개가 있을 수 있다
   이를 표로 나타내면 
   국어 영어 수학 평균
   n    m    k   z
   n'   m'   k'  z'
   ...
   이런식으로 되어 있을 것이다
   행은 그 열에 해당하는 점수가 들어가기 때문에
   행은 딱히 크게 상관이 없다 
   그래서 열우선 행무시 라는 용어를 사용함
   앞으로는 이렇게 1차함수임에도 입력이 여러개 들어온경우를 커버해본다
'''
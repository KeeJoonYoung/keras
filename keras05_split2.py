# train_test_split 을 이용한 데이터 분류

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

from sklearn.model_selection import train_test_split

# 데이터 
x = np.array(range(1, 101))
y = array(range(101, 201))

# train_test_split 함수로 validation 데이터를 구하려면
# train_test_split 함수를 두번쓰면 된다 (test 데이터를 반으로 나눔)
# 이 코드에서는 train : 60%, validation : 20%, test : 20% 를 하고 싶다면
# 아래와 같이 코드를 쓰면 된다 (처음에 test_size 0.4 로 잡고 이를 다시 절반씩 나눔)
# 항상 6 : 2 : 2 의 비율을 맞춰야 하는건 아니고 그냥 개발자 의도에 따라서 변경 가능
x_train, x_test, y_train, y_test = train_test_split( 
    x, y, train_size=0.6, test_size=0.4, shuffle=True
)

x_test, x_validation, y_test, y_validation = train_test_split(
    x_test, y_test, test_size=0.5, shuffle=True
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
model.fit(x_train, y_train, epochs=700, batch_size=1, validation_data=(x_validation, y_validation), validation_batch_size=1)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

predict = model.predict([101, 102, 103])
print(predict)
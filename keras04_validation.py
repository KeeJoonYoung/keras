import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([1, 2, 3, 4, 5])

x_validation = np.array([6, 7, 8])
y_validation = np.array([6, 7, 8])

x_test = np.array([9, 10, 11])
y_test = np.array([9, 10, 11])

# 모델
model = Sequential()
model.add(Dense(150, input_shape=(1, ), activation='relu')) # 활성화를 하면 성능이 더 좋아짐
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(1))

# 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_data=(x_validation, y_validation), validation_batch_size=1)

# 평가 및 예측
loss = model.evaluate(x_test, y_test) # x_test 와 y_test 는 실질적으로 훈련에 쓰이지 않기 때문에 버려지는 데이터라 볼 수 있다
print('loss : ', loss)

result = model.predict([9])
print('result : ', result)
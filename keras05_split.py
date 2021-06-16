from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

# 데이터 
x = np.array(range(1, 101))
y = array(range(101, 201))

x_train = x[:60]
x_validation = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_validation = y[60:80]
y_test = y[80:]

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

predict = model.predict([202])
print(predict) 
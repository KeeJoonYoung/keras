# mlp : multi layer perceptron

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 (국영수 과목에 연관없는 다른 과목을 추가한 형식?)
x = np.array([[10, 85, 70], [90, 85, 100], 
               [80, 50, 30], [43, 60, 100]]) # (4, 3)

y = np.array([75, 65, 33, 20]) # (4, )


# 모델 
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(50))
model.add(Dense(1))

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=700)

# 예상 및 결과
loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
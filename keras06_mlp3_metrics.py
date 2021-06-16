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
# metrics 는 실제 훈련시에 사용되는 지표가 아니라 참고용으로 다른 지표를 넣었을때 결과값이 어떻게 되는가를 보여준다
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x, y, batch_size=1, epochs=700)

# 예상 및 결과
loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)

'''
    accuracy 를 적용할때 지표 계산 방법
    : accuracy 는 정확도를 측정하는 계산법임
    최대 값이 1이고 0.8 이라면 80%가 일치하는 것
    (accuracy 대신에 acc 만 써도 된다)
    회귀 와 분류
    : 머신러닝에서 등장하는 분류는 지표로 accuracy 를 사용하지만 
    mae, mse 등은 회귀에서 주로 사용하는 지표가 된다 
    (분류에서도 쓰긴하지만 주된 용도로 따질때는 아님)
    정확도를 명확히 따져야 할때는 accuracy 를 쓰고
    그외에는 mae, mse 등을 사용 
'''

# R2 : 결정계수
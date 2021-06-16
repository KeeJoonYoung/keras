# output 이 2개 이상인 케이스에 대해서 코드 작성 (입력:출력 = n:m 인 경우)

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(100), range(301, 401), range(1, 101)]) # (3, 100)
y = np.array([range(711, 811), range(1, 101), range(201, 301)]) # (3, 100)

x = np.transpose(x) # (100, 3)
y = np.transpose(y) # (100, 3)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80, 3)  80% 를 훈련데이터로 나머지를 테스트 데이터로 뺐기 때문

# 모델 구성
model = Sequential()
model.add(Dense(90, input_shape=(3,))) 
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(3))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x_train, y_train, batch_size=1, epochs=300)

# 예상 및 결과
result = model.evaluate(x_test, y_test, batch_size=1)
print("result : ", result)

# 평가하는 값에 대한 예측을 해야하므로 x_test 를 넣어야 y 예측값이 나온다
y_predict = model.predict(x_test) # x 에 대한 예측은 x 의 결과값이 나올것
# y_predict 와 비교되는 것은 y_test 와 비교된다

from sklearn.metrics import mean_squared_error # mse 를 의미함

# y_test 는 처음에 주어진 결과 값, y_predict 는 evaluate 로 나온 예측값
# 주어진 평가값과 예측값 간의 차이를 계산하기 위해서 아래처럼 
# RMSE 와 MSE, R2 에 y_test, y_predict 를 넣은 것임
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

# y_predict 는 모델에 대한 예측 결과값을 의미한다

from sklearn.metrics import r2_score

# y_test : y 의 원래 값, y_predict : 예측 값
print('R2 : ', r2_score(y_test, y_predict)) # r2 가 높은건 그은 직선에 거의 인접해 있는 점들이 많다는 소리임
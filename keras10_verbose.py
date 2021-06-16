# verbose 옵션을 주는 방법에 대해 알아봄
# fit 에 verbose=0 을 주면 터미널상에 훈련의 과정이 나오지 않고 결과만 출력됨
# 터미널에 훈련 과정을 보여준다는 것은 CPU 의 속도보다 느리게 나올 수 있다는 의미가 되고
# 이는 속도 저하를 일으킬 수 있다는 의미가 된다
# 그래서 훈련 과정을 보이는게 필수가 아니라면 생략하는 것도 좋다

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(301, 401)]) # (5, 100)
y = np.array([range(711, 811), range(1, 101)]) # (2, 100)

x = np.transpose(x) # (100, 5)
y = np.transpose(y) # (100, 2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape) # (80, 5)
print(y_train.shape) # (80, 2)  80% 를 훈련데이터로 나머지를 테스트 데이터로 뺐기 때문

# 모델 구성
model = Sequential()
model.add(Dense(90, input_shape=(5,))) 
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(2))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x_train, y_train, batch_size=1, epochs=300, verbose=0)

'''
    verbose 의 값에 따른 변화
    verbose = 0 : 훈련 과정을 보이지 않는다
    verbose = 1 : 기본 값
    verbose = 2 : 훈련 진행시에 보여지는 progress bar 를 생략함 (하지만 훈련 과정은 보여줌)
    verbose = 3이상 : 훈련의 횟수만 보여주고 progress bar 나 metrics 등의 지표는 아무것도 안보임
 
'''

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
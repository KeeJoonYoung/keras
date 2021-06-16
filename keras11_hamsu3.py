# 함수형 모델 작성에 대해 알아본다 (1 : N 에 대한 함수형 모델)
# 모델의 두개 큰 축은 Sequential 과 함수형 모델 이고 다른 모델들은 이들을 섞어서 응용한것

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

y = np.array([range(100), range(711, 811)]) # (2, 100)
x = np.array(range(711, 811)) # (1, 100)

x = np.transpose(x) # (100, 1)
y = np.transpose(y) # (100, 2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

print(x_train.shape) # (80, 1)
print(y_train.shape) # (80, 2)  80% 를 훈련데이터로 나머지를 테스트 데이터로 뺐기 때문

# 모델 구성
input1 = Input(shape=(1,)) # input 이라는 별도의 layer 를 생성
dense1 = Dense(30)(input1) # hidden layer 의 첫번째 layer
dense2 = Dense(40)(dense1)
dense3 = Dense(50)(dense2) # 이전 레이어를 뒤에다 붙여서 늘리는 방식
dense4 = Dense(40)(dense3)
dense5 = Dense(80)(dense4)
output1 = Dense(2)(dense5) # 마지막 Dense 의 값은 output node 가 2개란 소리

# 함수형 모델은 Model 이란 객체 이름을 갖고 있다
model = Model(inputs=input1, outputs=output1)
# 함수형 모델은 시작 레이어와 끝 레이어를 명시해야하므로
# 레이어 설정이 끝난 시점에 선언해야 한다

model.summary()


# summary : 모델의 정보를 리턴
# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(2))
# model.summary() # 모델에 대한 정보를 출력한다
# 모델의 정보와 해당 모델이 갖는 레이어의 정보 그리고 param 정보를 보여준다
# output 에 None 이라 나오는 이유는 행을 무시했기 때문이다
# 2차원이 아닌 그 이상에서도 행 정보는 None 으로 나타난다
# 그리고 Sequential 모델에서는 input layer 에 대한 정보는 나오지 않는다

# summary 에서 param 값이 나오는 원리는 
# (레이어의 노드 갯수 + bias 노드 1개) * 다음 레이어의 노드 갯수 이다
# (위에서 입력 노드 5개 + bias 한개) * 다음 레이어 노드 3개 로 연산되서 18 이 나오고
# 그 다음도 동일한 연산으로 나오게 된다

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])
model.fit(x_train, y_train, batch_size=1, epochs=300, verbose=5)

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
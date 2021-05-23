import numpy as np
import tensorflow as tf

#1. 데이터 (= x와 y) 데이터는 항상 정제된 데이터를 넣는다.
x= np.array([1,2,3])
y= np.array([1,2,3])

#2. 모델 구성(인공 신경망)
from tensorflow.keras.models import Sequential #sequential=순서대로
from tensorflow.keras.layers import Dense #dense= y=wx+b

model = Sequential() #판을 깐거임
model.add(Dense(3, input_dim=1))  #dim=dimension 차원  결국 1개의 노드를 인풋하겠다는 의미 1개가 입력이 되서 3개가 나간다
model.add(Dense(4)) # 순차적이므로 윙에 있는게 자동으로 input이 된다. 즉 input이 3이다
model.add(Dense(2)) # input은 4이다. 왜냐하면 순차적이니까
model.add(Dense(1)) # layer을 추가하면서 node를 추가한것이다.
# 숫자는 node의 개수

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam') #mse 공부하기 (아마 오차들의 제곱의 합인듯) 당분간은 adam만 쓸거임
#위 과정이 loss의 최저를 찾는거임
model.fit(x,y, epochs=100, batch_size=1) # 배치 사이즈는 전체 사이즈를 얼마 잘라서 계산할까이다. 1로 하면 한번씩 계산 하는거다. 대이터개수= 배치사이즈*훈련회수 #에포는 훈련의 회수 여기선 100번이다.
# 숙제 배치사이즈를 명시 하지 않는다면? 돌아가긴함
# 훈련이 많을수록 좋다.
# 학습에 쓰인 데이터는 평가에 쓰면 안된다.

#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1)
print('loss :', loss) # mse값을 프린트한다.

results = model.predict([4]) # 4일때를 예측해봐라
print('result :', results)
 
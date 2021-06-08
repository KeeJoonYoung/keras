import numpy as np

#1. Data
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x[:5])
print(y[:5])
print(x.shape, y.shape) # (150, 4) (150,) 뒤에거가 (150,3)이 되고

print(dataset.feature_names)
print(dataset.DESCR) # 회귀 문제

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_test[0])

#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,)) # 라인 10의 4가 오고
h1 = Dense(10)(input1)
h2 = Dense(10)(h1)
h3 = Dense(10)(h2)
h4 = Dense(5)(h3)
output1 =  Dense(3, activation='softmax')(h4) # 3의 의미: 꽃의 종류가3개. 라인 10의 (150,3)에서 가져온다
model = Model(inputs=input1, outputs=output1)
# 회귀라면 Dense(1)(h4)가 맞다. 근데 지금은 분류이다. 분류는 [0,1,2]로 나오면 꽃 3개(첫번째,두번째,세번째)를 라벨링 한것이다. 머신은 두번째꽃*2=세번째꽃이라고 인식한다.
# 그래서 첫번째꽃을 [1,0,0] 두번째꽃을 [0,1,0] 세번째꽃을 [0,0,1]이라고 한다. 이것을  OneHotEncoding이라고 부른다. 1의 위치를 찾아내는것이다.
# y= [0,1,2,1]일때 y'=[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]으로 된다. (4, )에서 (4,3)으로 변했다. (데이터개수,분류종류=라벨의 개수)꼴

#3. Compile, Train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=5, epochs=100)

#4. Evaluate, Predict
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = model.predict(x_test)
print("input: ",x_test[:5])
print("GT: ", y_test[:5])
print("predict: ", y_predict[:5])

'''
    분류 모델은 소수점 조금이라도 다르면 다른 것으로 인식하기 때문에
    소수점을 반올림해서 동일한 대상으로 바꿔줘야할 필요가 있음
    이때 쓰이는게 활성화 함수 activation
    activation 은 값의 폭발을 막기위한 용도로 사용됨
    만약 y = wx + b ~ 에서 w 의 값이 너무 크면
    출력이 너무 크게 나와서 예측하려는 값과의 차이점이 너무 크게 됨
    따라서 w 의 값을 한정시키기 위한 용도로 쓰이는게 activation 이다
    (보통 0 ~ 1 사이의 소수로 지정함)
    가장 대표적인 activation 이 relu 이며
    Dense 인스턴스를 만들때 옵션으로 넣으면 된다
    다중 분류
    : 분류해야하는게 3개 이상일때 
    -> 다중 분류에서 출력 노드 activation 은 무조건 softmax 씀
    * activation 의 기본 값은 linear
    relu 는 0 ~ 1 사이의 값으로 지정하기 위해서 쓰이는 것
    loss 기준으로 쓰이는 것은 sparse_categorical_crossentropy 이다
'''
import numpy as np

#1. Data
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(x[:5])
# print(y[:5])
print(x.shape, y.shape) # (150, 4) (150,) \\\\\\\뒤에거가 (150,3)이 되고

## 원핫인코딩 OneHotEncoding
from tensorflow.keras.utils import to_categorical

print(y[145:]) # [2 2 2 2 2]
y= to_categorical(y) 
print(y[145:]) # 숫자를 안넣으면 끝까지이다 # [[0. 0. 1.] [0. 0. 1.] [0. 0. 1.] [0. 0. 1.] [0. 0. 1.]]
print(y[:5]) # [[1. 0. 0.] [1. 0. 0.] [1. 0. 0.] [1. 0. 0.] [1. 0. 0.]]
# print(y[:5])의 결과와 위의 두개를 비교
# 원핫인코딩은 (150,)를 (150,3)로 바꿔주는거다


# print(dataset.feature_names)
# print(dataset.DESCR) # 회귀 문제

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
h3 = Dense(10)(h2) # 명시는 안했지만 linear가 되어있다.
h4 = Dense(5)(h3)
output1 =  Dense(3, activation='softmax')(h4) # 3의 의미: 꽃의 종류가3개. 라인 10의 (150,3)에서 가져온다
model = Model(inputs=input1, outputs=output1)


#3. Compile, Train
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae']) # 'sparse_categorical_crossentropy'가 node의 값이 다양한 숫자가 나오면 1이나 0으로 만들어준다
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 
model.fit(x_train, y_train, batch_size=1, epochs=50)


#4. Evaluate, Predict
results = model.evaluate(x_test, y_test)
print('results : ', results)

# y_predict = model.predict(x_test)
# print("input: ",x_test[:5])
# print("GT: ", y_test[:5])
# print("predict: ", y_predict[:5])

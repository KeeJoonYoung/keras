import numpy as np
# sklearn.datasets: 교육용 데이터셋

#1
from sklearn.datasets import load_boston 
data = load_boston()
x = data.data
y = data.target # label 으로도 쓰기도함

print(x.shape) # (506, 13)
print(y.shape) # (506,) 집값에 대한 스칼라 1차원 벡터
print(data.feature_names)
print(data.DESCR) # Description 데이터셋에 대한 자세한 설명출력

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2 
input1 = Input(shape=(13,))
dense1 = Dense(30)(input1) 
dense2 = Dense(40)(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(50)(dense3) 
dense5 = Dense(30)(dense4)  
output1 = Dense(1)(dense5) 
 

model = Model(inputs=input1, outputs=output1)

#3
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=5)

#4
result = model.evaluate(x_test, y_test, batch_size=1)
print("result : ", result)

y_predict = model.predict(x_test) 

from sklearn.metrics import mean_squared_error, r2_score 

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print('R2 : ', r2_score(y_test, y_predict))
# fashion mnist :  6만개 28*28
# 그냥 mnist랑 크기 동일

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # 이미 나누어져있음

print(x_train.shape, x_test.shape) # (60000, 28, 28): 가로세로 28의 6만개 (10000, 28, 28)  # 28*28*1의 1은 생략, 흑백이라. 컬라라면 뒤에 적어줌
print(y_train.shape, y_test.shape) # (60000,) (10000,)

print(x_train[0])
print(y_train[0])

# 시각화
import matplotlib.pyplot as plt

plt.imshow(x_train[0], 'gray')
plt.show()

################ 똑같이 17_2,3을 만들어라

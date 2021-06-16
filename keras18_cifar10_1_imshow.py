# cifar10

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

print(x_train[0])
print(y_train[0])

# 시각화
import matplotlib.pyplot as plt

# plt.imshow(x_train[0], 'gray')
plt.imshow(x_train[1])
plt.show()

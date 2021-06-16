import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # 반드시 들어가야함. 정규화: 0~1로 데이터를 바꿔준다
    horizontal_flip=True, #수평으로 뒤집? 네
    vertical_flip=True, #수직으로 뒤집? 네
    width_shift_range=0.1, # 평행이동
    height_shift_range=0.1, # 평행이동
    rotation_range=5, #
    zoom_range=1.2, # 확대
    fill_mode='nearest'
)

test_datagen= ImageDataGenerator(
    rescale=1./255
) # 테스트 하는 데이터는 학습에는 쓰지 않으니까 평가에만 사용되니까 이거면 된다.

xy_train = train_datagen.flow_from_directory( # flow_from_directory란 어떻게 뽑아낼것이냐= 데이터화 된다.
    './tmp/horse-or-human',
    target_size=(300,300),
    batch_size=10000,
    class_mode='binary' # y가 bianary이다
)

xy_test = test_datagen.flow_from_directory(
    './tmp/testdata',
    target_size=(300,300),
    batch_size=5,
    class_mode='binary'
)

'''
결과
Found 1027 images belonging to 2 classes. train데이터 즉 1027,300,300,3
Found 256 images belonging to 2 classes. test데이터 즉 256,300,300,3
'''

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001C456108550>
# print(xy_train[0])
# 결과가 sklearn이랑 비슷하다. x데이터들, y데이터(5개= 배치사이즈를 의미) = (5,300,300,3)
# print(xy_train[0].shape) #는 에러 난다.
print(xy_train[0][0].shape) # (5, 300, 300, 3)
print(xy_train[0][1].shape) # (5,)
# print(xy_train[1][0].shape) # (5, 300, 300, 3)
# print(xy_train[1][1].shape) # (5,)
# # batch_size 자른만큼 생기는거다. 1027/5 = 205.4 => 206개
# print(xy_train[-1][0].shape) # (몰라, 300, 300, 3)
# print(xy_train[-1][1].shape) # (몰라,)


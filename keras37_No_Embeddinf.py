# 과제 36_3커피
# embedding을 빼라
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs=['너무 재밌어요', '참 최고에요','참 잘 만든 영화에요','추천하고 싶은 영화입니다.', 
        '한 번 더 보고 싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요','현욱이가 잘 생기긴 했어요']


# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
# 결과의 데이터의 shape가 모두 다르다.

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # post  # pad_sequences: 순서에 0을 채워 넣는다.
print(pad_x)
print(pad_x.shape) # (13, 5)로 데이터의 길이가 동일해졌다.

print(np.unique(pad_x)) # 독특한 놈들
print(len(np.unique(pad_x))) # 개수 28개

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=2, input_length=5)) # input_dim: 사전의 개수, 지금 개체의 개수가 28개이므로. 항상 최소개체 이상으로 넣어주어야 한다.
# model.add(Embedding(28,2))
# model.add(Conv1D(32,3))
# model.add(conv1D(32,3))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 7)              196    =   28*7 = input_dim=28 * output_dim=7
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5120
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 196
Trainable params: 196
Non-trainable params: 0
_________________________________________________________________

라인 33일때
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 2)              56
_________________________________________________________________
lstm (LSTM)                  (None, 32)                4480
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 4,569
Trainable params: 4,569
Non-trainable params: 0
_________________________________________________________________
라인 34일때
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 2)           56                명시하지 않아도 알아서 길이를 알아내서 해준다.
_________________________________________________________________
lstm (LSTM)                  (None, 32)                4480
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 4,569
Trainable params: 4,569
Non-trainable params: 0
_________________________________________________________________
'''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)




from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) # {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6} : 어절 순으로 자른다.
# word index순: 많이 나오는거 & 앞에 있는거 순으로 숫자가 주어진다.
x = token.texts_to_sequences([text])
print(x)    # [[3, 1, 1, 4, 5, 2, 2, 6]] : 숫자를 문자화 시킨다. 즉 라인6과 10은 같은거를 표현하는 것이다.

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)  # = 6
print(word_size) # = 6

x = to_categorical(x) # onehotincoding한 값으로 바꿔준다.

print(x) # [[[0. 0. 0. 1. 0. 0. 0.]  [0. 1. 0. 0. 0. 0. 0.]  [0. 1. 0. 0. 0. 0. 0.]  [0. 0. 0. 0. 1. 0. 0.]  [0. 0. 0. 0. 0. 1. 0.]  [0. 0. 1. 0. 0. 0. 0.]  [0. 0. 1. 0. 0. 0. 0.]  [0. 0. 0. 0. 0. 0. 1.]]]
print(x.shape) # (1, 8, 7) # 6이 아닌 7이 나온 이유는 맨 앞이 0으로 다 시작하는데 이건 to categorical 때문이다. 근데 우리는 1부터 시작하므로 0을 넣어주므로써 7이 된다.






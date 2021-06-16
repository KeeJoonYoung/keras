# 30_split 커피
import numpy as np

a = np.array(range(1,11)) #  뒤에꺼 하나 뺀거까지
size = 6
print(a) # [ 1  2  3  4  5  6  7  8  9 10]

def split_x(seq, size):
    aaa = []        
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) 
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)    
print(dataset)

x = dataset[:, :size-1] # 모든 행, 4번째 데이터까지 즉 4개의 열을 가져오겠다
y = dataset[:, size-1] # 4지정

print(x)
print(y)

x_pred = [[6,7,8,9,10]]

#2. 모델

#3. 컴파일, 훈련

#4. 평가, 예측

# 실습 Dense으로 만들기
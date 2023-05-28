import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('badword.csv',encoding='CP949')#.csv 파일을 Pandas를 이용하여 data에 저장
print('총 샘플의 수 :',len(data))
print(data[:5])#상위 5개의 샘플만 출력

X_data = data['v2']#본문은 X_data에 저장
y_data = data['v1']#1과 0의 값을 가진 레이블은 y_data에 저장
print('댓글 본문의 개수: {}'.format(len(X_data)))
print('레이블의 개수: {}'.format(len(y_data)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # 5823개의 행을 가진 X의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_data) # 단어를 숫자값, 인덱스로 변환하여 저장
print(sequences[:5])

word_to_index = tokenizer.word_index#케라스 토크나이저를 통해 토큰화와 정수 인코딩 과정을 수행
print(word_to_index)#정수가 어떤 단어에 부여되었는지 확인

vocab_size = len(word_to_index) + 1#단어 집합의 크기를 vocab_size에 저장.  패딩을 위한 토큰인 0번 단어를 고려하며 +1을 해서 저장
print('단어 집합의 크기: {}'.format((vocab_size)))

n_of_train = int(len(sequences) * 0.8)#데이터의 80%를 훈련용 데이터로 사용
n_of_test = int(len(sequences) - n_of_train)#20%를 테스트 데이터로 사용
print('훈련 데이터의 개수 :',n_of_train)
print('테스트 데이터의 개수:',n_of_test)

X_data = sequences
print('메일의 최대 길이 : %d' % max(len(l) for l in X_data))
print('메일의 평균 길이 : %f' % (sum(map(len, X_data))/len(X_data)))

max_len = 216
# 전체 데이터셋의 길이는 max_len으로 맞춥니다.
data = pad_sequences(X_data, maxlen = max_len)
print("훈련 데이터의 크기(shape): ", data.shape)

X_test = data[n_of_train:] #X_data 데이터 중에서 뒤의 1165개의 데이터만 저장
y_test = np.array(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 1165개의 데이터만 저장
X_train = data[:n_of_train] #X_data 데이터 중에서 앞의 4658개의 데이터만 저장
y_train = np.array(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 4658개의 데이터만 저장
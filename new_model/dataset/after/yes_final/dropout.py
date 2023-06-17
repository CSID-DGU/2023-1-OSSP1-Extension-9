import pandas as pd

# 데이터 로드
data = pd.read_csv("new_model/dataset/after/yes_final/all_data.csv", encoding='cp949')

# 라벨이 0인 데이터 추출
data_0 = data[data['label'] == 0]

# 라벨이 0이 아닌 데이터 추출
data_not_0 = data[data['label'] != 0]

# 라벨이 0인 데이터에서 40% 샘플링
sample_data_0 = data_0.sample(frac=0.4)

# 라벨이 0인 샘플링된 데이터와 라벨이 0이 아닌 데이터 결합
new_data = pd.concat([sample_data_0, data_not_0])

# 새로운 데이터를 csv 파일로 저장
new_data.to_csv("all_data_final.csv", index=False)

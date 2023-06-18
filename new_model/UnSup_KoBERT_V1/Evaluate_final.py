import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# csv 파일 로드
df = pd.read_csv('new_model/UnSup_KoBERT_V1/test_Unsup_Kobert_V1.csv', encoding='cp949')

# softmax 값 중 가장 큰 값을 가진 레이블을 선택하여 예측 레이블로 사용
df['predicted_label'] = df[['혐오', '성차별', '일베', 'Unknown']].idxmax(axis=1)

# 레이블 이름을 숫자로 변환 (Unknown_softmax -> -1, label0_softmax -> 0, ...)
label_dict = {'혐오': 0, '성차별': 1, '일베': 2, 'Unknown': 3}
df['predicted_label'] = df['predicted_label'].map(label_dict)

# 예측 레이블과 실제 레이블 비교하여 평가 지표 계산
accuracy = accuracy_score(df['label'], df['predicted_label'])
precision = precision_score(df['label'], df['predicted_label'], average='macro')
recall = recall_score(df['label'], df['predicted_label'], average='macro')
f1 = f1_score(df['label'], df['predicted_label'], average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

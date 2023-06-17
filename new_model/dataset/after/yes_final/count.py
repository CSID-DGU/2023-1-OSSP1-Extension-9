import pandas as pd

data = pd.read_csv("new_model/dataset/after/yes_final/all_data_final.csv")

label_counts = data['label'].value_counts()

print(label_counts)
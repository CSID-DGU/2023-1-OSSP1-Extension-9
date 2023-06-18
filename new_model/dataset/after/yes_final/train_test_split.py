from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


# Load and split the dataset
dataset = pd.read_csv('new_model/dataset/after/yes_final/all_data_final.csv')
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
val_dataset.to_csv('test_data.csv', index=False)
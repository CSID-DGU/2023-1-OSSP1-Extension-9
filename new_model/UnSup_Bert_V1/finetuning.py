from transformers import BertConfig, BertTokenizer, BertModel, get_scheduler
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig.from_json_file("new_model/UnSup_Bert_V1/model_outputs/config.json")
model_state_dict = torch.load("new_model/UnSup_Bert_V1/model_outputs/pytorch_model.bin")
model = BertModel(config)
model.load_state_dict(model_state_dict, strict=False)

tokenizer = BertTokenizer.from_pretrained("new_model/UnSup_Bert_V1/model_outputs")

# Define custom dataset class
class SentenceDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __getitem__(self, index):
        text = self.data['comment'][index]
        label = self.data['label'][index]
        encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length',
                                             max_length=128, truncation=True, return_tensors='pt')
        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)

# Load and split the dataset
dataset = SentenceDataset('new_model/dataset/after/all_data.csv')
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)   

num_epochs = 10
optimizer = AdamW(model.parameters(), lr=2e-5)

loss_fn = torch.nn.CrossEntropyLoss()

model.to(device)

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]  # Assuming BERT model has a pooler layer
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

model.eval()

val_loss = 0.0
val_accuracy = 0.0

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]  # Assuming BERT model has a pooler layer
        loss = loss_fn(logits, labels)
        val_loss += loss.item()

        _, predicted_labels = torch.max(logits, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()
        val_accuracy += correct_predictions

val_loss /= len(val_dataset)
val_accuracy /= len(val_dataset)

model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

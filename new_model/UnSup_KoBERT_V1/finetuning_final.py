from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import pandas as pd
from tokenization_kobert import KoBertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained("new_model/UnSup_KoBERT_V1/model_outputs", num_labels=3)
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

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
dataset = SentenceDataset('new_model/dataset/after/no_final/all_data_final.csv')
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Move model to device
model.to(device)

# Set model to train mode
model.train()

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    train_accuracy = 0.0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        train_loss += loss.item()

        _, predicted_labels = torch.max(logits, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()
        train_accuracy += correct_predictions

        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataset)
    train_accuracy /= len(train_dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print("--------------------")      

    # Validation loop
    model.eval()
    
    val_loss = 0.0
    val_accuracy = 0.0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            _, predicted_labels = torch.max(logits, dim=1)
            correct_predictions = (predicted_labels == labels).sum().item()
            val_accuracy += correct_predictions
            val_loss /= len(val_dataset)
            val_accuracy /= len(val_dataset)

model.save_pretrained("fine_tuned_model")
print("Complete fine-tuning!!!")
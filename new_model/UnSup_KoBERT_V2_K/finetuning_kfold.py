from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import KFold
import pandas as pd
from tokenization_kobert import KoBertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load the dataset
dataset = SentenceDataset('new_model/dataset/after/yes_final/all_data_final.csv')

# Perform K-fold cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold+1}/{num_folds}")

    # Split dataset into train and validation
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Define the model
    model = BertForSequenceClassification.from_pretrained("new_model/UnSup_KoBERT_V2/model_outputs", num_labels=4)
    model.to(device)

    # Define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        model.train()

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

        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
        print("--------------------")

    # Save the model
    model_path = f"fine_tuned_model_fold_{fold+1}"
    model.save_pretrained(model_path)
    print(f"Model for Fold {fold+1} saved: {model_path}")

print("Complete fine-tuning!!!")

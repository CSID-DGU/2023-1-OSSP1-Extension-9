from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tokenization_kobert import KoBertTokenizer

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("new_model/UnSup_KoBERT_V1/fine_tuned_model", num_labels=3)
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the evaluation dataset from CSV
dataset = pd.read_csv("new_model/dataset/after/all_data.csv")
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Define a function to tokenize and encode the input text
def preprocess(text):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].squeeze()
    attention_mask = encoded_text['attention_mask'].squeeze()
    return input_ids, attention_mask

# Prepare the evaluation data
eval_inputs = [preprocess(text) for text in val_dataset['comment']]
eval_labels = val_dataset['label'].tolist()

# Evaluate the model
model.to(device)
model.eval()

predictions = []

with torch.no_grad():
    for input_ids, attention_mask in eval_inputs:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).squeeze().item()
        predictions.append(predicted_labels)

eval_accuracy = accuracy_score(eval_labels, predictions)

print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
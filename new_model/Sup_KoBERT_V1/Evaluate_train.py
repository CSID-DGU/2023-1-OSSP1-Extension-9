from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tokenization_kobert import KoBertTokenizer

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("new_model/Sup_KoBERT_V1/fine_tuned_model", num_labels=3)
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the evaluation dataset from CSV
eval_data = pd.read_csv("new_model/dataset/after/all_data.csv")

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
eval_inputs = [preprocess(text) for text in eval_data['comment']]
eval_labels = eval_data['label'].tolist()

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
eval_precision = precision_score(eval_labels, predictions, average='weighted')
eval_recall = recall_score(eval_labels, predictions, average='weighted')
eval_f1_score = f1_score(eval_labels, predictions, average='weighted')

print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
print(f"Evaluation Precision: {eval_precision:.4f}")
print(f"Evaluation Recall: {eval_recall:.4f}")
print(f"Evaluation F1-score: {eval_f1_score:.4f}")
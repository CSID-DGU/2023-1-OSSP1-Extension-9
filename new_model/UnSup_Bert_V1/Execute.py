from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("new_model/UnSup_Bert_V1/fine_tuned_model", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("new_model/UnSup_Bert_V1/fine_tuned_model")

device = torch.device("cpu")

sentence = "아 기분 딱좋노 이기야 ㅋㅋㅋ 홍어 멸망"
encoded_input = tokenizer.encode_plus(
    sentence,
    add_special_tokens=True,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1)
    print(logits)
    
class_labels = ['혐오', '성차별', '일베']  # Replace with your actual class labels
predicted_class = class_labels[predicted_labels.item()]
print(predicted_class)
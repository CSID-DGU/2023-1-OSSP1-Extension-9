from transformers import BertConfig, BertModel, BertForSequenceClassification
import os

config = BertConfig.from_pretrained("monologg/kobert", num_labels=4)
model = BertForSequenceClassification.from_pretrained(os.path.join("new_model/Sup_KoBERT_V2", "best_checkpoint.pt"), config=config)

model.save_pretrained("model_outputs")

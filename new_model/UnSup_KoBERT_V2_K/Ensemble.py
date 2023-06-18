import torch
from transformers import BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved models
model_paths = [
    "fine_tuned_model_fold_1",
    "fine_tuned_model_fold_2",
    "fine_tuned_model_fold_3",
    "fine_tuned_model_fold_4",
    "fine_tuned_model_fold_5"
]

models = []
for path in model_paths:
    model = BertForSequenceClassification.from_pretrained(path, num_labels=4)
    model.to(device)
    model.eval()
    models.append(model)

# Ensemble the models
ensemble_model = BertForSequenceClassification.from_pretrained(model_paths[0], num_labels=4)
ensemble_model.to(device)

with torch.no_grad():
    for ensemble_param, *model_params in zip(ensemble_model.parameters(), *[model.parameters() for model in models]):
        ensemble_param.copy_(torch.mean(torch.stack([param.data for param in model_params]), dim=0))

# Save the ensemble model
ensemble_model_path = "ensemble_model"
ensemble_model.save_pretrained(ensemble_model_path)
print(f"Ensemble model saved: {ensemble_model_path}")

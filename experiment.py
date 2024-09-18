import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import numpy as np

# Load the pretrained RoBERTa model and tokenizer
model_name = 'roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the SST-2 dataset from the GLUE benchmark
dataset = load_dataset("glue", "sst2")
eval_data = dataset['validation']

print('eval_data')

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

tokenized_eval_data = eval_data.map(preprocess_function, batched=True)

# Convert to DataLoader for batching
def collate_fn(batch):
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return input_ids, attention_mask, labels

eval_dataloader = DataLoader(tokenized_eval_data, batch_size=16, collate_fn=collate_fn)

# Perform inference and collect predictions
model.eval()
predictions, true_labels = [], []

for batch in eval_dataloader:
    input_ids, attention_mask, labels = [x.to(device) for x in batch]
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
    
    predictions.extend(preds.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy of RoBERTa on SST-2: {accuracy:.4f}")

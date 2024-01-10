import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from data.MovieDataset import MovieDataset
from model.BERTClassifier import BERTClassifier
from utils.Utils import train, evaluate

# Set up parameters
bert_model_name = 'bert-base-cased'
num_classes = 5
max_length = 512
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = MovieDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = MovieDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
torch.cuda.empty_cache()

best_accuracy = 0.0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    if accuracy > best_accuracy:
      torch.save(model.state_dict(), "bert_classifier.pth")
      best_accuracy = accuracy
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
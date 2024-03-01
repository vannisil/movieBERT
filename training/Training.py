import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from data.MovieDataset import MovieDataset
from model.BERTClassifier import BERTClassifier
from utils.Utils import train, evaluate

# Set up parameters
bert_model_name = 'bert-base-cased'  # Initializing BERT model name
num_classes = 5  # Number of classes for classification
max_length = 512  # Maximum length of input sequences
batch_size = 16  # Batch size for training
num_epochs = 5  # Number of epochs for training
learning_rate = 2e-5  # Learning rate for optimizer

# Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Create training and validation datasets
train_dataset = MovieDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = MovieDataset(val_texts, val_labels, tokenizer, max_length)

# Create dataloaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Set device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT-based classifier model
model = BERTClassifier(bert_model_name, num_classes).to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Calculate total number of training steps for scheduler
total_steps = len(train_dataloader) * num_epochs

# Initialize scheduler with warmup steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Clear GPU cache
torch.cuda.empty_cache()

# Initialize best accuracy for model checkpointing
best_accuracy = 0.0

# Start training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train the model
    train(model, train_dataloader, optimizer, scheduler, device)

    # Evaluate the model on validation set
    accuracy, report = evaluate(model, val_dataloader, device)

    # Check if current accuracy is better than the previous best accuracy
    if accuracy > best_accuracy:
        # Save the model if current accuracy is better
        torch.save(model.state_dict(), "bert_classifier.pth")
        best_accuracy = accuracy

    # Print validation accuracy and evaluation report
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
import torch.nn.functional as F

def train(model, data_loader, optimizer, scheduler, device):
    # Set the model to train mode
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Calculate the loss
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    # Set the model to evaluation mode
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device, dtype=torch.long)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get the predicted labels
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    # Calculate accuracy and classification report
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def prediction(text, model, tokenizer, device, max_length=512):
    # Set the model to evaluation mode
    model.eval()

    # Tokenize the input text and create input tensors
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Disable gradient computation during prediction
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get the predicted labels
        _, preds = torch.max(outputs, dim=1)

    return preds, probabilities

from torch import nn
from transformers import BertModel

"BERT model with dropout set to 0 and two fully connected layers"
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        # Initializing BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Dropout layer to prevent overfitting (set to 0)
        self.dropout = nn.Dropout(0.0)
        # Fully connected layer 1
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        # Fully connected layer 2
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extracting pooled output
        pooled_output = outputs.pooler_output
        # Applying dropout
        x = self.dropout(pooled_output)
        # Passing through first fully connected layer
        logits1 = self.fc1(x)
        # Passing through second fully connected layer
        logits2 = self.fc2(logits1)
        return logits2

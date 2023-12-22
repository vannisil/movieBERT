import torch
from transformers import BertTokenizer

from model.BERTClassifier import BERTClassifier
from utils.Utils import prediction

bert_model_name = 'bert-base-cased'
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load('./model/bert_classifier.pth', map_location=device))
model.eval()

test_text =  "\"The Notebook\" - A heartwarming love story based on Nicholas Sparks' novel, portraying the enduring romance between a young couple despite the challenges they face."
prediction = prediction(test_text, model, tokenizer, device)
tags = ["murder", "romantic", "violence", "psychedelic", "comedy"]

for i in range(0,len(tags)):
  if prediction == i:
    tag = tags[i]
    print(f"This film is: {tag}")
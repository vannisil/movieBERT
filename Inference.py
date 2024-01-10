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


test_text = ["\"Titanic\" is a 1997 epic romance and disaster film directed by James Cameron. The story revolves around the ill-fated maiden voyage of the RMS Titanic in 1912. The film follows the romance between Jack Dawson, a penniless artist played by Leonardo DiCaprio, and Rose DeWitt Bukater, an upper-class woman engaged to a wealthy industrialist, played by Kate Winslet. The love story unfolds against the backdrop of the luxurious but doomed ocean liner. As the ship collides with an iceberg and tragedy strikes, Jack and Rose must navigate the chaos and danger to survive.", "\"Rocky\" is a 1976 American sports drama film written and starring Sylvester Stallone. The film follows the story of Rocky Balboa, a small-time boxer from Philadelphia, who gets a shot at the world heavyweight championship. Despite being an underdog, Rocky seizes the opportunity to train rigorously and face the reigning champion, Apollo Creed, in a match that becomes a symbol of determination and the human spirit.", "\"Airplane!\" (1980), directed by Jim Abrahams and the Zucker brothers. The movie is a spoof of disaster films and follows the story of Ted Striker, a former fighter pilot, who must overcome his fear of flying to save a flight full of passengers when the crew falls ill due to food poisoning. Packed with absurd and slapstick humor, \"Airplane!\" is known for its rapid-fire jokes, visual gags, and memorable one-liners, making it a classic in the genre of parody comedy."]
tags = ["murder", "romantic", "violence", "psychedelic", "comedy"]
for i in range(len(test_text)):
  print(f"{i+1}) {test_text[i]}")
  predicted_tags = prediction(test_text[i], model, tokenizer, device)
  for j in range(0,len(tags)):
    if predicted_tags == j:
      tag = tags[j]
      print(f"    The best tag for this film is: {tag}")
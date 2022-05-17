import json
import torch
import random
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as f:
  intents = json.load(f)
  intents = intents["intents"]

FILE_NAME = "data.pth"
data = torch.load(FILE_NAME)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
state = data["state"]

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
model.load_state_dict(state)
model.eval()

customer_service_rep_name = "Anya"
while True:
  sentence = input("How can I help you? type 'quit' to exit. \n")
  
  if sentence == 'quit':
    break

  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  output = model(X)
  _, predicted = torch.max(output, dim=1)
  idx = predicted.item()

  # softmax converts to a set of probabilities
  probs = torch.softmax(output, dim=1)
  prob = probs[0][idx]

  if prob > 0.80:
    tag = tags[idx]
    for intent in intents:
      if intent["tag"] == tag:
        print(f"{customer_service_rep_name}: {random.choice(intent['answers'])}")
  else:
    print(f"{customer_service_rep_name}: I don't understand...")

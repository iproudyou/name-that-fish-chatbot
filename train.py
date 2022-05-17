import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet 

with open('intents.json', 'r') as f:
  intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag)
  for question in intent['questions']:
    tokenized = tokenize(question)
    all_words.extend(tokenized)
    xy.append((tokenized, tag))

ignore_words = ['.', ',', '?', '!']
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for tokenized, tag in xy:
  bow = bag_of_words(tokenized, all_words)
  X_train.append(bow)

  idx = tags.index(tag)
  Y_train.append(idx)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
  def __init__(self):
    self.n_samples = len(X_train)
    self.x_train = X_train
    self.y_train = Y_train

  def __getitem__(self, idx):
    return self.x_train[idx], self.y_train[idx]
  
  def __len__(self):
    return self.n_samples

batch_size = 8
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for (words, labels) in train_loader:
    words = words.to(device)
    labels = labels.to(device)

    outputs = model(words)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if (epoch + 1) % 100 == 0:
    print(f"epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss={loss.item():.4f}")

data = {
  "input_size": input_size,
  "hidden_size": hidden_size, 
  "output_size": output_size,
  "all_words": all_words,
  "tags": tags,
  "state": model.state_dict()
}

FILE_NAME = 'data.pth'
torch.save(data, FILE_NAME)

print(f"file saved to {FILE_NAME}")

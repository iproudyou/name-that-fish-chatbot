import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
  bow = np.zeros(len(all_words), dtype=np.float32)
  tokenized_sentence = [stem(word) for word in tokenized_sentence]

  for idx, word in enumerate(all_words):
    if word in tokenized_sentence:
      bow[idx] = 1.0

  return bow

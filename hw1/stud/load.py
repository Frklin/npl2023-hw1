import torch
from torch.utils.data import Dataset, DataLoader
import jsonlines
import config
import numpy as np
from hw1.stud.utils import preprocess_sentence
import nltk
from nltk.tag import pos_tag

nltk.download('averaged_perceptron_tagger')

class MyDataset(Dataset):
    def __init__(self, path, word2idx, label2idx, pos2idx, char2idx):
        self.data = []
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.pos2idx = pos2idx
        self.char2idx = char2idx
        self.pos_tags = []
        self.load_data(path)

    def load_data(self, path):
        with jsonlines.open(path) as f:
            print('Loading data from {}'.format(path))
            for line in f:
                tokens = preprocess_sentence(line['tokens'])
                labels = (line['labels'])
                self.data.append((tokens, labels))

                if config.POS:
                    pos = pos_tag(tokens)
                    pos = [tag[1] for tag in pos]
                    self.pos_tags.append([self.pos2idx.get(tag, 37) for tag in pos])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        pos = self.pos_tags[idx] if config.POS else None
        # chars = [[self.char2idx.get(char.lower(), 1) for char in token] for token in tokens]
        # pos = [self.pos2idx.get(pos_tag(tokens)[i][1],37) for i in range(len(tokens))]
        tokens  = [self.word2idx.get(token, self.word2idx[config.UNK_TOKEN]) for token in tokens]
        labels = [self.label2idx[label] for label in labels]
        return tokens, labels, pos#, chars
    


import torch
from torch.utils.data import Dataset, DataLoader
import jsonlines
import config
import numpy as np
# import toeknizer




class Dataset(Dataset):
    def __init__(self, path, word2idx, label2idx):
        self.data = []
        self.word2idx = word2idx
        self.label2idx = label2idx
        with jsonlines.open(path) as f:
            for line in f:
                tokens = line['tokens']
                labels = line['labels']
                self.data.append((tokens, labels))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        tokens  = [self.word2idx.get(token, self.word2idx[config.UNK_TOKEN]) for token in tokens]
        labels = [self.label2idx[label] for label in labels]
        return tokens, labels
    

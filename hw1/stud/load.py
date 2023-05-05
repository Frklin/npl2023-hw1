import torch
from torch.utils.data import Dataset
import jsonlines
import config
import numpy as np
from utils import preprocess_sentence
import nltk
from nltk.tag import pos_tag

nltk.download('averaged_perceptron_tagger')



class MyDataset(Dataset):
    '''
    A PyTorch dataset for processing tokenized text data.

    Args:
        path (str): The path to the data file in JSON Lines format.
        word2idx (dict): A dictionary mapping words to unique indices.
        label2idx (dict): A dictionary mapping labels to unique indices.
        pos2idx (dict): A dictionary mapping parts-of-speech tags to unique indices.
        char2idx (dict): A dictionary mapping characters to unique indices.
    '''

    def __init__(self, path, word2idx, label2idx, pos2idx):
        self.data = []  # List of tuples, where each tuple is a (sentence, label) pair
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.pos2idx = pos2idx
        self.pos_tags = []  # List of lists, where each sublist is a sequence of POS tags for a sentence
        self.chars = []  # List of lists of lists, where each sublist is a sequence of character indices for a token in a sentence
        self.load_data(path)

    def load_data(self, path):
        '''
        Load tokenized data from a JSON Lines file.

        Args:
            path (str): The path to the data file in JSON Lines format.
        '''
        with jsonlines.open(path) as f:
            print('Loading data from {}'.format(path))
            for line in f:
                tokens = preprocess_sentence(line['tokens'])  # Tokenize and preprocess the sentence
                labels = line['labels']  # Get the corresponding label sequence
                self.data.append((tokens, labels))
                if config.CHAR: 
                    char_sent = []
                    for token in tokens:
                        char_sent.append([ord(char) if ord(char)<199 else 200 for char in token])
                    self.chars.append(char_sent)
                if config.POS:
                    pos = pos_tag(tokens)  # Tag the sentence with POS tags
                    pos = [tag[1] for tag in pos]  # Get just the POS tags
                    self.pos_tags.append([self.pos2idx[tag] for tag in pos])

    def __len__(self):
        '''
        Return the number of sentences in the dataset.
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Get a single item (sentence) from the dataset.

        Args:
            idx (int): The index of the sentence to retrieve.

        Returns:
            tuple: A tuple containing the token indices, label indices, POS tag indices, and character indices (if CHAR=True).
        '''
        tokens, labels = self.data[idx]

        chars = self.chars[idx] if config.CHAR else None
        tokens = [self.word2idx.get(token, self.word2idx[config.UNK_TOKEN]) for token in tokens]  # Convert the tokens to indices
        labels = [self.label2idx[label] for label in labels]  # Convert the labels to indices
        pos = self.pos_tags[idx] if config.POS else None

        return tokens, labels, pos, chars
    






    
import config
import random
import os
import re
import numpy as np
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def seed_everything(seed = config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    tokens, labels, pos, chars = zip(*batch)
    token_lengths = [len(t) for t in tokens]
    max_length = max(token_lengths)
    #find the longest word in the batch
    max_word_length = max([len(word) for sent in chars for word in sent]) if config.CHAR else None
    # padded_tokens = pad_sequence(tokens, batch_first=True, padding_value = config.PAD_IDX)
    # padded_labels = pad_sequence(labels, batch_first=True, config.PAD_VAL)

    padded_tokens = [t + [config.PAD_IDX] * (max_length - len(t)) for t in tokens]
    padded_labels = [l + [config.PAD_VAL] * (max_length - len(l)) for l in labels]
    padded_pos = [p + [config.PAD_IDX] * (max_length - len(p)) for p in pos] if config.POS else None
    # padded_chars = [[char + [config.PAD_IDX]*(max_word_length-len(char)) for char in sen] + [[config.PAD_IDX]*max_word_length]*max_length for sen in chars] if config.CHAR else None
    padded_chars = [[(sen[i] + [config.PAD_IDX]*(max_word_length-len(sen[i]))) if i < len(sen) else ([config.PAD_IDX] * max_word_length)  for i in range(max_length)] for sen in chars] if config.CHAR else None

    tokens_tensor = torch.LongTensor(padded_tokens)
    labels_tensor = torch.LongTensor(padded_labels)
    lengths_tensor = torch.LongTensor(token_lengths)
    pos_tensor = torch.LongTensor(padded_pos) if config.POS else None
    char_tensor = torch.LongTensor(padded_chars) if config.CHAR else None
    
    return tokens_tensor, labels_tensor, lengths_tensor, pos_tensor, char_tensor

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()




def preprocess_sentence(tokens):

    # 1. Convert to lower case
    tokens = [word.lower() for word in tokens]
    # 2. Remove special characters and digits
    # tokens = [re.sub(r'[^a-zA-Z\s]', '', word) for word in tokens]

    # 4. Remove stopwords
    # tokens = [token if token not in stopwords else "<SW>" for token in tokens ]

    # 5. Lemmatize tokens
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens
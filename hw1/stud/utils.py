import config
import numpy as np
import os
import random
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def collate_fn(batch):
    '''
    Collate a batch of data for training.

    Args:
        batch (list of tuple): A list of tuples representing a batch of data. Each tuple contains four elements:
            1. A list of token ids.
            2. A list of label ids.
            3. A list of part-of-speech tag ids (optional).
            4. A list of character ids (optional).

    Returns:
        tuple: A tuple containing five elements:
            1. A tensor representing the token ids of the batch.
            2. A tensor representing the label ids of the batch.
            3. A tensor representing the lengths of each token sequence in the batch.
            4. A tensor representing the part-of-speech tag ids of the batch (optional).
            5. A tensor representing the character ids of the batch (optional).
    '''
    # Unpack the batch into lists of tokens, labels, part-of-speech tag ids, and character ids
    tokens_list, labels_list, pos_list, chars_list = zip(*batch)

    # Get the length of each token sequence in the batch
    token_lengths = [len(tokens) for tokens in tokens_list]

    # Get the maximum length of any token sequence in the batch
    max_length = max(token_lengths)

    # Get the maximum length of any word in the batch (if character-level modeling is enabled)
    max_word_length = max([len(word) for sentence in chars_list for word in sentence]) if config.CHAR else None

    # Pad the token, label, part-of-speech tag, and character sequences to the maximum length
    padded_tokens = [tokens + [config.PAD_IDX] * (max_length - len(tokens)) for tokens in tokens_list]
    padded_labels = [labels + [config.PAD_VAL] * (max_length - len(labels)) for labels in labels_list]
    padded_pos = [pos + [config.PAD_IDX] * (max_length - len(pos)) for pos in pos_list] if config.POS else None
    padded_chars = [[(sentence[i] + [config.PAD_IDX]*(max_word_length-len(sentence[i]))) if i < len(sentence) else ([config.PAD_IDX] * max_word_length) for i in range(max_length)] for sentence in chars_list] if config.CHAR else None

    # Convert the padded sequences into PyTorch tensors
    tokens_tensor = torch.LongTensor(padded_tokens)
    labels_tensor = torch.LongTensor(padded_labels)
    lengths_tensor = torch.LongTensor(token_lengths)
    pos_tensor = torch.LongTensor(padded_pos) if config.POS else None
    char_tensor = torch.LongTensor(padded_chars) if config.CHAR else None
    
    return tokens_tensor, labels_tensor, lengths_tensor, pos_tensor, char_tensor


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


def seed_everything(seed = config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
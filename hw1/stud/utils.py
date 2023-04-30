import config
import random
import os
import numpy as np
import torch

def seed_everything(seed = config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    tokens, labels = zip(*batch)
    token_lengths = [len(t) for t in tokens]
    max_length = max(token_lengths)

    # padded_tokens = pad_sequence(tokens, batch_first=True, padding_value = config.PAD_IDX)
    # padded_labels = pad_sequence(labels, batch_first=True, config.PAD_VAL)
    padded_tokens = [t + [config.PAD_IDX] * (max_length - len(t)) for t in tokens]
    padded_labels = [l + [config.PAD_VAL] * (max_length - len(l)) for l in labels]

    tokens_tensor = torch.LongTensor(padded_tokens)
    labels_tensor = torch.LongTensor(padded_labels)
    lengths_tensor = torch.LongTensor(token_lengths)
    # pad_index = 0

    # return sentences_pad, labels_pad, lengths_tensor

    return tokens_tensor, labels_tensor, lengths_tensor


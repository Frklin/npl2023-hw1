import numpy as np
import pandas as pd
from word2vec import Word2Vec, DatasetGenerator
from typing import List
from bilstm import BiLSTM
from embeddings import load_embeddings
from utils import preprocess_sentence
import config
import torch
from nltk.tag import pos_tag

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    train = pd.read_json("data/train.csv")

    data = DatasetGenerator(train)
    return RandomBaseline()


class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self):
        embeddings, self.word2idx = load_embeddings()
        self.model = BiLSTM(embeddings=embeddings, label_count=config.LABEL_COUNT).to(config.DEVICE)

        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE)))

        self.pos2idx = {pos: i for i, pos in enumerate(config.POS_TAGS)}

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        pos_tags = []
        chars = []
        char_inputs = None
        pos_inputs = None

        for i, sentence in enumerate(tokens):
            for j, word in enumerate(preprocess_sentence(sentence)):
                tokens[i][j] = self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])
        
        if config.POS:
            pos = pos_tag(tokens)
            pos = [tag[1] for tag in pos]
            pos_tags.append([self.pos2idx[tag] for tag in pos])
        tokens = [torch.tensor(sentence, dtype=torch.long) for sentence in tokens]

        if config.CHAR: 
            char_sent = []
            for token in tokens:
                char_sent.append([ord(char) if ord(char)<199 else 200 for char in token])
                chars.append(char_sent)

        # Pad sentences
        inputs = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.word2idx[config.PAD_TOKEN])
        if config.CHAR:
            char_inputs = torch.nn.utils.rnn.pad_sequence(char_sent, batch_first=True, padding_value=config.PAD_IDX)
        if config.POS:
            pos_inputs = torch.nn.utils.rnn.pad_sequence(pos_tags, batch_first=True, padding_value=config.PAD_IDX)

        predictions = self.model(inputs, char_inputs, pos_inputs)

        predictions = [sentence[:len(tokens[i])] for i, sentence in enumerate(predictions)]
        
        return predictions

import sys
sys.path.append('hw1/stud/')
sys.path.append('hw1')
import numpy as np
from typing import List
from bilstm import BiLSTM
from embeddings import load_embeddings
from utils import preprocess_sentence
import config
import torch
import nltk
from nltk.tag import pos_tag
import torch.nn.functional as F
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    
    return StudentModel()


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
        # self.model = BiLSTM(embeddings=embeddings, label_count=config.LABEL_COUNT, device=config.DEVICE).to(config.DEVICE)

        # self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE)))
        self.model = torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE))

        self.pos2idx = {pos: i for i, pos in enumerate(config.pos2idx)}

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        pos_tags = []
        chars = []
        char_inputs = None
        pos_vectors = None
        length_list = torch.LongTensor([len(sent) for sent in tokens], device=config.DEVICE)
        max_length = max(length_list)
        max_word_length = max([len(word) for sent in tokens for word in sent])

        if config.POS:
            for sentence in tokens:
                pos = pos_tag(sentence)
                pos = [tag[1] for tag in pos]

                pos_tags.append([self.pos2idx[tag] for tag in pos])
            padded_pos = [p + [config.PAD_IDX] * (max_length - len(p)) for p in pos_tags] if config.POS else None

            pos_vectors = torch.zeros((len(padded_pos), max_length, config.POS_DIM),dtype=torch.float32).to(config.DEVICE)
                
            for i, sen in enumerate(pos_tags):
                for j, tag in enumerate(sen):
                    pos_vectors[i][j] = F.one_hot(torch.tensor(tag), num_classes=config.POS_DIM)


        if config.CHAR: 
            for sentence in tokens:
                char_sent = []
                for token in sentence:
                    char_sent.append([ord(char) if ord(char)<199 else 200 for char in token])
                chars.append(char_sent)
                
            padded_chars = [[(sen[i] + [config.PAD_IDX]*(max_word_length-len(sen[i]))) if i < len(sen) else ([config.PAD_IDX] * max_word_length)  for i in range(max_length)] for sen in chars] if config.CHAR else None
            char_inputs = torch.tensor(padded_chars, dtype=torch.long, device=config.DEVICE) if config.CHAR else None

        for i, sentence in enumerate(tokens):
            for j, word in enumerate(preprocess_sentence(sentence)):
                tokens[i][j] = self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])

        tokens = [torch.tensor(sentence, dtype=torch.long, device=config.DEVICE) for sentence in tokens]

        # Pad sentences
        inputs = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.word2idx[config.PAD_TOKEN]).to(config.DEVICE)

        # print(inputs)
        # print(char_inputs)
        # print(pos_vectors)

        predictions = self.model.decode(inputs, length_list, pos_vectors, chars = char_inputs)

        predictions = [sentence[:len(tokens[i])] for i, sentence in enumerate(predictions)]

        predictions = [[config.idx2label[pred] for pred in sentence] for sentence in predictions]
        
        return predictions


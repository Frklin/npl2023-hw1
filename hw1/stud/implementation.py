
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
        self.embeddings = load_embeddings()
        self.word2idx = config.word2idx
        self.model = BiLSTM(embeddings=self.embeddings, label_count=config.LABEL_COUNT, device=config.DEVICE).to(config.DEVICE)

        # self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE)))
        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(config.DEVICE)))

        self.model.eval()

        self.pos2idx = {pos: i for i, pos in enumerate(config.pos2idx)}

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        pos_tags = []
        chars = []
        inputs = []

        # if config.POS:
        #     for sentence in tokens:
        #         pos = pos_tag(sentence)
        #         pos = [tag[1] for tag in pos]

        #         pos_tags.append([self.pos2idx[tag] for tag in pos])
        #     padded_pos = [p + [config.PAD_IDX] * (max_length - len(p)) for p in pos_tags] if config.POS else None

        #     pos_vectors = torch.zeros((len(padded_pos), max_length, config.POS_DIM),dtype=torch.float32).to(config.DEVICE)

        #     for i, sen in enumerate(pos_tags):
        #         for j, tag in enumerate(sen):
        #             pos_vectors[i][j] = F.one_hot(torch.tensor(tag), num_classes=config.POS_DIM)


        # if config.CHAR:
        #     for sentence in tokens:
        #         char_sent = []
        #         for token in sentence:
        #             char_sent.append([ord(char) if ord(char)<199 else 200 for char in token])
        #         chars.append(char_sent)

        #     padded_chars = [[(sen[i] + [config.PAD_IDX]*(max_word_length-len(sen[i]))) if i < len(sen) else ([config.PAD_IDX] * max_word_length)  for i in range(max_length)] for sen in chars] if config.CHAR else None
        #     char_inputs = torch.tensor(padded_chars, dtype=torch.long, device=config.DEVICE) if config.CHAR else None
        length_list = [len(sentence) for sentence in tokens]
        max_length = max(length_list)
        max_word_length = max([len(word) for sentence in tokens for word in sentence])
        
        for sentence in tokens:
            sentence = preprocess_sentence(sentence)
            char_sent = []
            for word in sentence:
                char_sent.append([ord(char) if (ord(char)<199 and ord(char) != 0) else 200 for char in word])
            chars.append(char_sent)
            pos = pos_tag(sentence)  
            pos = [tag[1] for tag in pos] 
            pos_tags.append([self.pos2idx[tag] for tag in pos])
            inputs.append([self.word2idx.get(word, self.word2idx[config.UNK_TOKEN]) for word in sentence])
        
      
        padded_inputs = [sen + [config.word2idx[config.PAD_TOKEN]] * (max_length - len(sen)) for sen in inputs]
        padded_chars = [[(sen[i] + [0]*(max_word_length-len(sen[i]))) if i < len(sen) else ([0] * max_word_length)  for i in range(max_length)] for sen in chars] 
        padded_pos = [p + [0] * (max_length - len(p)) for p in pos_tags] 

        inputs = torch.tensor(padded_inputs, dtype=torch.long, device=config.DEVICE)
        char_inputs = torch.tensor(padded_chars, dtype=torch.long, device=config.DEVICE) 
        # pos_vectors = torch.tensor(pos_vectors, dtype=torch.long, device=config.DEVICE)
        pos_vectors = torch.zeros((len(padded_pos), max_length, config.POS_DIM),dtype=torch.float32).to(config.DEVICE)

        for i, sen in enumerate(padded_pos):
            for j, tag in enumerate(sen):
                pos_vectors[i][j] = F.one_hot(torch.tensor(tag), num_classes=config.POS_DIM)

        # for i, sentence in enumerate(tokens):
        #     for j, word in enumerate(preprocess_sentence(sentence)):
        #         tokens[i][j] = self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])

        # tokens = [torch.tensor(sentence, dtype=torch.long, device=config.DEVICE) for sentence in tokens]

        # Pad sentences
        # inputs = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.word2idx[config.PAD_TOKEN]).to(config.DEVICE)
        
        # inputs, length_list, pos_vectors, char_inputs = self.pad_everything(inputs, pos_tags, chars)
        # if config.CLASSIFIER == "crf":
        m = (inputs != config.word2idx[config.PAD_TOKEN])
        mask = m.clone().detach().to(torch.uint8)
        predictions = self.model.decode(inputs, length_list, pos_vectors, char_inputs, mask)
        # else:
        #     predictions = self.model(inputs, length_list, pos_vectors, chars = char_inputs)
        #     predictions = torch.argmax(predictions, dim=2).detach().cpu().numpy().tolist()

        # predictions = [sentence[:len(tokens[i])] for i, sentence in enumerate(predictions)]

        predictions = [[config.idx2label[pred] for pred in sentence] for sentence in predictions]

        # plot_confusion_matrix
        return predictions


    def pad_everything(self, tokens: List[List[str]], pos_tags: List[List[str]], chars: List[List[List[str]]]):
        token_lengths = [len(token) for token in tokens]
        max_length = max(token_lengths)
        max_word_length = max([len(word) for sentence in chars for word in sentence]) 
        pad_token = config.word2idx[config.PAD_TOKEN]
        pad_pos = config.pos2idx[config.PAD_TOKEN]

        padded_tokens = [token + [pad_token] * (max_length - len(token)) for token in tokens]
        padded_pos = [pos + [pad_pos] * (max_length - len(pos)) for pos in pos_tags] if config.POS else None
        padded_chars = [[(sentence[i] + [0]*(max_word_length-len(sentence[i]))) if i < len(sentence) else ([0] * max_word_length) for i in range(max_length)] for sentence in chars]
        
        tokens_tensor = torch.LongTensor(padded_tokens)
        lengths_tensor = torch.LongTensor(token_lengths)
        pos_tensor = torch.LongTensor(padded_pos) 
        char_tensor = torch.LongTensor(padded_chars)
   
        return tokens_tensor, lengths_tensor, pos_tensor, char_tensor

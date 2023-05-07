
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


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # Initialize empty lists
        pos_tags = []
        chars = []
        inputs = []

        # Get the maximum length of sentences and maximum length of words
        length_list = [len(sentence) for sentence in tokens]
        max_length = max(length_list)
        max_word_length = max([len(word) for sentence in tokens for word in sentence])
        
        # Preprocess each sentence and convert words to character-level embeddings
        for sentence in tokens:
            sentence = preprocess_sentence(sentence)
            char_sent = [[ord(char) if (ord(char) < 199 and ord(char) != 0) else 200 for char in word] for word in sentence]
            chars.append(char_sent)
            pos = pos_tag(sentence)  
            pos = [tag[1] for tag in pos] 
            pos_tags.append([config.pos2idx[tag] for tag in pos])
            inputs.append([self.word2idx.get(word, self.word2idx[config.UNK_TOKEN]) for word in sentence])
        
        # Pad the input sequences and convert character-level embeddings to tensors
        padded_inputs = [sen + [config.word2idx[config.PAD_TOKEN]] * (max_length - len(sen)) for sen in inputs]
        padded_chars = [[(sen[i] + [0]*(max_word_length-len(sen[i]))) if i < len(sen) else ([0] * max_word_length)  for i in range(max_length)] for sen in chars] 
        padded_pos = [p + [config.pos2idx[config.PAD_TOKEN]] * (max_length - len(p)) for p in pos_tags] 

        inputs = torch.tensor(padded_inputs, dtype=torch.long, device=config.DEVICE)
        char_inputs = torch.tensor(padded_chars, dtype=torch.long, device=config.DEVICE)         
        pos_tensor = torch.tensor(padded_pos, dtype=torch.long, device=config.DEVICE)
        pos_vectors = torch.zeros((len(padded_pos), max_length, config.POS_DIM),dtype=torch.float32).to(config.DEVICE)

        # Convert POS tags to one-hot vectors
        for i, sen in enumerate(pos_tensor):
            for j, tag in enumerate(sen):
                pos_vectors[i][j] = F.one_hot(tag, num_classes=config.POS_DIM)

        # Create a mask to ignore padding tokens
        m = (pos_tensor != config.pos2idx[config.PAD_TOKEN])
        mask = m.clone().detach().to(torch.uint8)

        # Make predictions using the model
        predictions = self.model.decode(inputs, length_list, pos_vectors, None, mask)
        predictions = [[config.idx2label[pred] for pred in sentence] for sentence in predictions]

        return predictions

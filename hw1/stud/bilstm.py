import sys
sys.path.append('hw1/stud')
sys.path.append('hw1')
import torch
import torch.nn as nn
import config
from crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(self, embeddings, label_count, device, hidden_size=config.HIDDEN_SIZE, lstm_layers=config.N_LSTMS, dropout=config.DROPRATE, classifier=config.CLASSIFIER):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad = True  
        self.classifier = classifier
        self.hidden_dim = hidden_size
        self.device = device

        # LSTM
        self.bilstm = nn.LSTM(embeddings.shape[1], hidden_size, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.hidden = None
        
        # Softmax
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, label_count)
        self.dropout = VariationalDropout(dropout)  
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size)

        # CRF
        self.crf = CRF(label_count)

    def init_hidden(self, batch_size):
        return (
            torch.randn(4, batch_size, self.hidden_dim).to(self.device),
            torch.randn(4, batch_size, self.hidden_dim).to(self.device),
        )
    

    def forward(self, tokens, token_lengths, mask=None, flag=1):
        self.hidden = self.init_hidden(tokens.shape[0])
        x = self.embeddings(tokens)

        x = pack_padded_sequence(x, token_lengths, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.bilstm(x, self.hidden)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        if self.classifier == 'CRF' and flag:
            # mask = [i != config.PAD_IDX for i in tokens]
            print("sono entrato nel crf")
            score, path = self.crf.decode(x, mask=mask)
            return score, path

 
        return x

    def loss(self, x, y, token_lengths, mask=None):
        emissions = self.forward(x, token_lengths, mask=mask, flag=0)
        # x = self.embeddings(x)
        # emissions = self.bilstm(x)[0]
        nll = self.crf(emissions, y, mask=mask)
        return nll







class VariationalDropout(nn.Module):
    def __init__(self, dropout_rate):
        super(VariationalDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training:
            return x

        mask = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = mask / (1 - self.dropout_rate)
        mask = mask.expand_as(x)

        return x * mask
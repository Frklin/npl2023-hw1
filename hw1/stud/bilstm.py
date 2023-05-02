import sys
sys.path.append('hw1/stud')
sys.path.append('hw1')
import torch
import torch.nn as nn
import config
# from crf import CRF
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class BiLSTM(nn.Module):
    def __init__(self, embeddings, label_count, device, hidden_size=config.HIDDEN_SIZE, lstm_layers=config.N_LSTMS, dropout=config.DROPRATE, classifier=config.CLASSIFIER):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad = True
        self.classifier = classifier
        self.hidden_dim = hidden_size // 2
        self.device = device
        self.n_filters = 30
        # compute embedding dimension with char embeddings and pos tags if applicable
        self.embed_dim = config.EMBEDDING_SIZE
        if config.CHAR:
            self.embed_dim += self.n_filters
        if config.POS:
            self.embed_dim += config.POS_DIM

        # LSTM
        self.bilstm = nn.LSTM(self.embed_dim, hidden_size // 2, num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.hidden = None
        
        # Softmax
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, label_count)
        self.dropout = VariationalDropout(dropout)  
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.softmax = nn.Softmax(dim=-1)

        # CRF
        self.crf = CRF(label_count, batch_first=True)

        # CNN
        self.char_embeddings = nn.Embedding(config.CHAR_VOCAB_SIZE, config.CHAR_DIM)
        self.conv1 = nn.Conv1d(config.CHAR_DIM, self.n_filters, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, padding=1)

    def init_hidden(self, batch_size):
        return (
            torch.randn(4, batch_size, self.hidden_dim).to(self.device),
            torch.randn(4, batch_size, self.hidden_dim).to(self.device),
        )
    

    def forward(self, tokens, token_lengths, pos=None, chars=None, mask=None):
        self.hidden = self.init_hidden(tokens.shape[0])
        x = self.embeddings(tokens)
        if config.CHAR:
            chars = chars.view(-1, chars.shape[-1])
            char_embs = self.char_embeddings(chars)
            # char_embs = char_embs.permute(0, 1, 3, 2)
            char_embs = torch.einsum('ijk->ikj', char_embs)
            char_cnn = self.relu(self.conv1(char_embs))
            char_cnn = self.maxpool1(char_cnn)
            char_cnn = torch.max(char_cnn, 2)[0]
            char_cnn = char_cnn.view(tokens.shape[0], tokens.shape[1], -1)
            x = torch.cat((x, char_cnn), dim=2)
            print(x.shape)
        if config.POS:
            x = torch.cat((x, pos), dim=-1)
            print(x.shape)
        print(x.shape)

        x = pack_padded_sequence(x, token_lengths, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.bilstm(x, self.hidden)
        x, _ = pad_packed_sequence(x, batch_first=True)
        # x = self.norm1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear(x)
        # x = self.norm2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        # x = self.softmax(x)
        

        return x

    def loss(self, x, y, token_lengths, pos=None, chars=None, mask=None):
        emissions = self.forward(x, token_lengths, pos=pos, chars=chars, mask=mask)
        nll = -self.crf(emissions, y, mask=mask, reduction='token_mean')
        return nll

    def decode(self, x, token_lengths,pos=None, chars=None, mask=None):
        emissions = self.forward(x, token_lengths, pos=pos, chars=chars, mask=mask)
        preds = self.crf.decode(emissions, mask=mask)
        return preds





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
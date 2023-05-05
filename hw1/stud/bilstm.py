import sys
sys.path.append('hw1/stud')
sys.path.append('hw1')
import torch
import torch.nn as nn
import config
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    '''
    BiLSTM (also BiLSTM-CRF and BiLSTM-CNN) model .
    '''

    def __init__(self, embeddings, label_count, device=config.DEVICE, hidden_size=config.HIDDEN_SIZE, lstm_layers=config.N_LSTMS, dropout=config.DROPRATE, classifier=config.CLASSIFIER):
        super(BiLSTM, self).__init__()
        
        # Embedding layer
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.embeddings.weight.requires_grad = False
        
        # Classifier type
        self.classifier = classifier
        
        # Dimension of hidden layer
        self.hidden_dim = hidden_size // 2
        
        # Number of LSTM layers
        self.lstm_layers = lstm_layers

        # Device (CPU or GPU)
        self.device = device
        
        # Number of filters in CNN layer
        self.n_filters = config.CNN_FILTERS
        
        # Compute embedding dimension with character embeddings and POS tags if applicable
        self.embed_dim = config.EMBEDDING_SIZE
        if config.CHAR:
            self.embed_dim += self.n_filters
        if config.POS:
            self.embed_dim += config.POS_DIM

        # LSTM layer
        self.bilstm = nn.LSTM(self.embed_dim, hidden_size // 2, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.hidden = self.init_hidden(config.BATCH_SIZE)
        
        # Softmax layer
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, label_count)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # CRF layer
        self.crf = CRF(label_count, batch_first=True)

        # CNN layer
        self.char_embeddings = nn.Embedding(config.CHAR_VOCAB_SIZE, config.CHAR_DIM)
        self.conv1 = nn.Conv1d(config.CHAR_DIM, self.n_filters, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, padding=1)

    def init_hidden(self, batch_size):
        '''
        Initialize hidden state of LSTM layer.
        '''
        return (
            torch.randn(2 * self.lstm_layers, batch_size, self.hidden_dim),
            torch.randn(2 * self.lstm_layers, batch_size, self.hidden_dim),
        )
    
    def forward(self, tokens, token_lengths, pos=None, chars=None, mask=None):
        """
        Forward pass of the model.

        Args:
            tokens (torch.Tensor): Tensor of token indices, shape (batch_size, seq_len).
            token_lengths (torch.Tensor): Tensor of token lengths, shape (batch_size,).
            pos (torch.Tensor or None, optional): Tensor of POS tag indices, shape (batch_size, seq_len) or None.
            chars (torch.Tensor or None, optional): Tensor of character indices, shape (batch_size, seq_len, max_word_len) or None.
            mask (torch.Tensor or None, optional): Tensor of masks, shape (batch_size, seq_len) or None.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len, label_count) representing the unnormalized scores for each
                possible label at each position.
        """
        # Embedding layer
        x = self.embeddings(tokens)

        # Character-level embeddings
        if config.CHAR:
            chars = chars.view(-1, chars.shape[-1])
            char_embs = self.char_embeddings(chars)
            char_embs = torch.einsum('ijk->ikj', char_embs)
            char_cnn = self.relu(self.conv1(char_embs))
            char_cnn = self.maxpool1(char_cnn)
            char_cnn = torch.max(char_cnn, 2)[0]
            char_cnn = char_cnn.view(tokens.shape[0], tokens.shape[1], -1)
            x = torch.cat((x, char_cnn), dim=2)

        # POS tag embeddings
        if config.POS:
            x = torch.cat((x, pos), dim=-1)

        # LSTM layer
        x = pack_padded_sequence(x, token_lengths, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.bilstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=config.PAD_IDX)

        # Dense layers
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
        
    def loss(self, x, y, token_lengths, pos=None, chars=None, mask=None):
        '''Compute the negative log-likelihood loss of the model on the given inputs.

        Args:
        - x (torch.Tensor): Input tokens of shape (batch_size, max_seq_len)
        - y (torch.Tensor): Target labels of shape (batch_size, max_seq_len)
        - token_lengths (torch.Tensor): Lengths of sequences in x, of shape (batch_size,)
        - pos (torch.Tensor): Part-of-speech tags of shape (batch_size, max_seq_len, pos_dim) if using POS features, None otherwise.
        - chars (torch.Tensor): Character indices of shape (batch_size, max_seq_len, max_word_len) if using char features, None otherwise.
        - mask (torch.Tensor): Mask of valid positions in x, of shape (batch_size, max_seq_len)

        Returns:
        - nll (torch.Tensor): Negative log-likelihood loss value, a scalar.
        '''
        emissions = self.forward(x, token_lengths, pos=pos, chars=chars, mask=mask)
        nll = -self.crf(emissions, y, mask=mask, reduction='token_mean')
        return nll

    def decode(self, x, token_lengths, pos=None, chars=None, mask=None):
        '''Decode the best label sequence for the given inputs using Viterbi decoding.

        Args:
        - x (torch.Tensor): Input tokens of shape (batch_size, max_seq_len)
        - token_lengths (torch.Tensor): Lengths of sequences in x, of shape (batch_size,)
        - pos (torch.Tensor): Part-of-speech tags of shape (batch_size, max_seq_len, pos_dim) if using POS features, None otherwise.
        - chars (torch.Tensor): Character indices of shape (batch_size, max_seq_len, max_word_len) if using char features, None otherwise.
        - mask (torch.Tensor): Mask of valid positions in x, of shape (batch_size, max_seq_len)

        Returns:
        - preds (List[List[int]]): List of predicted label sequences, one for each input sequence in x.
        '''
        emissions = self.forward(x, token_lengths, pos=pos, chars=chars, mask=mask)
        preds = self.crf.decode(emissions, mask=mask)
        return preds

    def unfreeze(self):
        '''Unfreeze the pre-trained word embeddings to allow them to be fine-tuned during training.'''
        for param in self.embeddings.parameters():
            param.requires_grad = True










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
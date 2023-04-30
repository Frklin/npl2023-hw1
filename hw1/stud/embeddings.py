import numpy as np
import config
import torch
from gensim.models import KeyedVectors



def load_embeddings(embedding_path: str = config.EMBEDDINGS_PATH):
  
    word2vec = KeyedVectors.load(embedding_path) 

    word2idx = word2vec.key_to_index
    embeddings = word2vec.vectors

    word2idx['<PAD>'] = config.PAD_IDX
    word2idx['<UNK>'] = len(word2idx)-1

    embeddings[config.PAD_IDX] = np.zeros((1, word2vec.vector_size), dtype=np.float32)
    # embeddings = np.append(embeddings, np.zeros((1, word2vec.vector_size), dtype=np.float32), axis=0)
    embeddings = np.append(embeddings, np.random.rand(1, word2vec.vector_size).astype(np.float32), axis=0)
    
    return torch.tensor(embeddings, dtype=torch.float32), word2idx



class Word2Vec:
    def __init__(self) -> None:
        pass



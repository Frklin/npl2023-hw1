import numpy as np
import config
import torch
from gensim.models import KeyedVectors
import gensim.downloader as api
import os 



def load_embeddings(embedding_path: str = config.EMBEDDINGS_PATH, embedding_type: str = config.EMBEDDING_MODEL):
    
    if not os.path.exists(embedding_path):
        print("Embedding file not found. Downloading...")
        emb_file = api.load(embedding_type)
        emb_file.save(embedding_path)
        print("Embedding file saved to {}".format(embedding_path))
    else:
        emb_file = KeyedVectors.load(embedding_path) 

    word2idx = emb_file.key_to_index
    embeddings = emb_file.vectors

    word2idx['<PAD>'] = config.PAD_IDX
    word2idx['<UNK>'] = len(word2idx)-1

    embeddings[config.PAD_IDX] = np.zeros((1, emb_file.vector_size), dtype=np.float32)
    # embeddings = np.append(embeddings, np.zeros((1, word2vec.vector_size), dtype=np.float32), axis=0)
    embeddings = np.append(embeddings, np.random.rand(1, emb_file.vector_size).astype(np.float32), axis=0)
    
    return torch.tensor(embeddings, dtype=torch.float32), word2idx



class Word2Vec:
    def __init__(self) -> None:
        pass



import numpy as np
import config as config
import torch
from gensim.models import KeyedVectors
import gensim.downloader as api
import os 



def load_embeddings(embedding_path: str = config.EMBEDDINGS_PATH, embedding_type: str = config.EMBEDDING_MODEL):
    '''
    Load pre-trained word embeddings.

    Args:
        embedding_path (str): Path to the pre-trained embeddings file.
        embedding_type (str): Type of the pre-trained embeddings (e.g. GenW2V, GenGlove, Fasttext).

    Returns:
        tuple: A tuple containing:
            - embeddings (tensor): A tensor containing the pre-trained word embeddings.
            - word2idx (dict): A dictionary mapping words to their corresponding indices.
    '''
    # Check if the file exists
    if not os.path.exists(embedding_path):
        print("Embedding file not found. Downloading...")
        
        # Load the pre-trained embeddings based on the specified type
        if embedding_type == 'GenW2V':
            gensim_embeddings = api.load('word2vec-google-news-300')
        elif embedding_type == 'GenGlove':
            gensim_embeddings = api.load('glove-wiki-gigaword-300')
        elif embedding_type == 'Fasttext':
            gensim_embeddings = api.load('fasttext-wiki-news-subwords-300')

        gensim_embeddings.save(embedding_path)
        print("Embedding file saved to {}".format(embedding_path))
    else:
        gensim_embeddings = KeyedVectors.load(embedding_path)

    # Create a dictionary mapping words to their corresponding indices
    word2idx = gensim_embeddings.key_to_index

    # Get the pre-trained word embeddings
    embeddings = gensim_embeddings.vectors

    # Add special tokens to the word2idx dictionary
    word2idx['<PAD>'] = config.PAD_IDX
    word2idx['<UNK>'] = len(word2idx) - 1

    # Reassign the indices for the old idx 0
    word2idx['the'] = 40000

    # Add embeddings for special tokens
    embeddings[40000] = embeddings[0]
    embeddings[config.PAD_IDX] = np.zeros((1, gensim_embeddings.vector_size), dtype=np.float32)
    embeddings = np.append(embeddings, np.random.rand(1, gensim_embeddings.vector_size).astype(np.float32), axis=0)

    # Convert the embeddings to a tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    return embeddings_tensor, word2idx
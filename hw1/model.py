from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from seqeval.metrics import f1_score as f1
from seqeval.scheme import IOB2




class Model:
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """
        A simple wrapper for your model

        Args: tokens: list of list of strings. The outer list represents the sentences, the inner one the tokens
        contained within it. Ex: [ ["Hard", "Rock", "Hell", "III", "."], ["It", "was", "the", "largest", "naval",
        "battle", "in", "Western", "history", "."] ]

        Returns:
            list of list of predictions associated to each token in the respective position.
            Ex: Ex: [ ["O", "O", "O", "O", "O"], ["O", "O", "O", "O", "O", "B-ACTION", "O", "O", "O", "O"] ]

        """
        raise NotImplementedError


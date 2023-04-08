import numpy as np
import config
import pandas as pd
import torch 

class Word2Vec:

    def __init__(self,dataset,V):
        self.dataset = dataset
        self.embeddings = torch.rand(len(V),config.EMBEDDING_SIZE)
        self.context = torch.rand(len(V),config.EMBEDDING_SIZE)
        self.V = V

    def train(self):
        pass

    def train_one_epoch(self,word):
        neighbors = self.dataset[word]
        actual_neighbors = neighbors[:,1] == 1

        print("actual neighbors: ",actual_neighbors)

        for i,neighbor in enumerate(actual_neighbors):
            negatives = neighbors[i:i+config.NEG_SAMPLES,0]
            negatives_targets = neighbors[i:i+config.NEG_SAMPLES,1]
            input_word = self.embeddings[self.V[word]]
            context_words = self.context[self.V[negatives]]
            print("word: ", word, "neighbor: ",neighbor)
            print("input word: ",input_word)
            print("negatives: ",negatives)
            print("context words: ",context_words)
            print("context words shape: ",context_words.shape)
            print("input word shape: ",input_word.shape)

            predictions = torch.sigmoid(torch.matmul(input_word,context_words.T))
            print("predictions: ",predictions)
            errors = negatives_targets - predictions
            print("errors: ",errors)

            self.embeddings[self.V[word]] += torch.matmul(errors,context_words)
            self.context[self.V[negatives]] += torch.matmul(errors,input_word)

    def predict(self):
        pass



class DatasetGenerator:

    def __init__(self,dataset):
        self.V = dict()
        self.dataframe = dict()
        self.window = config.SLIDING_WINDOW
        self.neg_samples = config.NEG_SAMPLES
        self.dataset = dataset

    def generate(self):

    
        for sentence in self.dataset.tokens:
            for word in sentence:
                if word not in self.V:
                    self.V[word] = 1
                else:
                    self.V[word] += 1

        for sentence in self.dataset.tokens:
            for i,word in enumerate(sentence):
                for k in range(1,self.window+1):
                    if i-k >= 0:
                        self.dataframe.append((word,sentence[i-k]),1)             # (word,context) or all together? (word, [])
                    if i+k < len(sentence):
                        self.dataframe.append((word,sentence[i+k]),1)
                    for j in range(self.neg_samples):
                        self.dataframe.append((word,np.random.choice(self.V.keys())),0)
        

    def getVocab(self):
        return self.V
    
    def getDataset(self):
        return self.dataframe
    


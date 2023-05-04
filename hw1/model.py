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



class Trainer:
    def __init__(self, model, train_dataloader, dev_dataloader, optimizer, loss_function, device ,clip = 0, classifier = config.CLASSIFIER):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        
        self.clip = clip
        self.classifier = classifier
        
        #Early stopping
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience = 3 
        self.epochs_without_improvement = 0
        self.patience = 10


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        y_true_train = []
        y_pred_train = []

        pbar = tqdm(self.train_dataloader, total=20000//config.BATCH_SIZE)

        for tokens, labels, token_lengths, pos, chars in pbar:
            tokens, labels = tokens.to(self.device), labels.to(self.device)
            pos,chars = pos.to(self.device) if config.POS else None, chars.to(self.device) if config.CHAR else None

            if config.POS:
                pos_vectors = torch.zeros((len(pos), torch.max(token_lengths), config.POS_DIM),dtype=torch.float32).to(self.device)
                
                for i, sen in enumerate(pos):
                    for j, tag in enumerate(sen):
                        pos_vectors[i][j] = F.one_hot(tag, num_classes=config.POS_DIM)
            else:
                pos_vectors = None

            self.optimizer.zero_grad()

            if self.classifier == 'softmax':

                logits = self.model(tokens, token_lengths, pos_vectors, chars)

                loss = self.loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))

                preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                labels = labels.view(-1).cpu().numpy()
                org_idxs = np.where(labels != config.PAD_VAL)[0]
                labels = labels[org_idxs]
                preds = preds[org_idxs].tolist()
            
            elif self.classifier == 'crf':

                m = (labels != config.PAD_VAL)
                mask = m.clone().detach().to(torch.uint8)
                
                # self.model.zero_grad()
                loss = self.model.loss(tokens, labels, token_lengths, pos_vectors, chars, mask)

                preds = self.model.decode(tokens,token_lengths, pos_vectors, chars, mask)
                labels = labels.view(-1).cpu().numpy()
                org_idxs = np.where(labels != config.PAD_VAL)[0]
                labels = labels[org_idxs]
                preds = sum(preds, [])

            else:
                raise NotImplementedError
            
            if self.clip != 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            loss.backward()
            self.optimizer.step()

            y_true_train.extend(labels.tolist())
            y_pred_train.extend(preds)

            total_loss += loss.item()



        train_loss = total_loss / len(self.train_dataloader)
        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro') 
        # train_seq_f1 = f1(y_true_train, y_pred_train, mode='strict', scheme=IOB2, average='macro')
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted')





        return train_loss, train_accuracy, train_f1, train_precision, train_recall

    def evaluate(self):
        self.model.eval()
        y_true_val = []
        y_pred_val = []
        total_loss = 0

        with torch.no_grad():
            for tokens, labels, token_lengths, pos, chars in tqdm(self.dev_dataloader):
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                pos, chars = pos.to(self.device) if config.POS else None, chars.to(self.device) if config.CHAR else None

                if config.POS:
                    pos_vectors = torch.zeros((len(pos), torch.max(token_lengths), config.POS_DIM),dtype=torch.float32).to(self.device)
                    
                    for i, sen in enumerate(pos):
                        for j, tag in enumerate(sen):
                            pos_vectors[i][j] = F.one_hot(tag, num_classes=config.POS_DIM)
                else:
                    pos_vectors = None


                if self.classifier == 'crf':
                    m = (labels != config.PAD_VAL)
                    mask = m.clone().detach().to(torch.uint8)
                    loss = self.model.loss(tokens, labels, token_lengths, pos_vectors, chars, mask)

                    preds = self.model.decode(tokens,token_lengths, pos_vectors, chars, mask)
                    labels = labels.view(-1).cpu().numpy()
                    org_idxs = np.where(labels != config.PAD_VAL)[0]
                    labels = labels[org_idxs]
                    preds = sum(preds, [])

                elif self.classifier == 'softmax':
                    logits = self.model(tokens, token_lengths, pos_vectors, chars)
                    loss = self.loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))
                    preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                    labels = labels.view(-1).cpu().numpy()
                    orig_idxs = np.where(labels != config.PAD_VAL)[0]
                    labels = labels[orig_idxs]
                    preds = preds[orig_idxs].tolist()

                total_loss += loss.item()

                y_true_val.extend(labels.tolist())
                y_pred_val.extend(preds)


        val_loss = total_loss / len(self.dev_dataloader)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        # val_seq_f1 = f1(y_true_val, y_pred_val,average="macro", mode="strict", scheme=IOB2)
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted')


        return val_loss, val_accuracy, val_f1, val_precision, val_recall

    def train(self, num_epochs):
        best_f1 = 0
        schedluer = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        for epoch in range(num_epochs):
            if epoch >= config.UNFREEZE_EPOCH:
                self.model.unfreeze()
                print("Unfreezing the model")
            train_loss, train_accuracy, train_f1, train_precision, train_recall = self.train_epoch()
            dev_loss, dev_accuracy, dev_f1, dev_precision, dev_recall = self.evaluate()
            print(f"Epoch {epoch} train_loss: {train_loss}, train_accuracy: {train_accuracy}, train_F1-score: {train_f1}")
            print(f"Epoch {epoch} val_loss: {dev_loss}, val_accuracy: {dev_accuracy}, val_F1-score: {dev_f1}")
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_f1_score": train_f1,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "val_loss": dev_loss,
                "val_accuracy": dev_accuracy,
                "val_f1_score": dev_f1,
                "val_precision": dev_precision,
                "val_recall": dev_recall
            })
            schedluer.step(dev_loss)

            if dev_loss < self.best_val_loss:
                best_f1 = dev_f1
                self.best_val_loss = dev_loss
                torch.save(self.model.state_dict(), config.MODEL_PATH)
                print(f"Saving model at epoch {epoch+1} with validation loss: {self.best_val_loss}.")
                self.epochs_without_improvement = 0

            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                dev_loss = self.best_val_loss
                dev_f1 = best_f1
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {self.best_val_loss}.")
                break



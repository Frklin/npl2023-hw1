from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm



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
    def __init__(self, model, train_dataloader, dev_dataloader, optimizer, loss_function, device ,clip = False, classifier = config.CLASSIFIER):
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
        self.patience = 1  
        self.epochs_without_improvement = 0


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        y_true_train = []
        y_pred_train = []

        pbar = tqdm(self.train_dataloader, total=20000//config.BATCH_SIZE)

        for idx, (tokens, labels, token_lengths) in enumerate(pbar):
            tokens, labels = tokens.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.classifier == 'softmax':

                logits = self.model(tokens, token_lengths)

                loss = self.loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))

                preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                labels = labels.view(-1).cpu().numpy()
            



            
            elif self.classifier == 'crf':

                mask = (labels != config.PAD_VAL).float()
                
                self.model.zero_grad()
                loss = self.model.loss(tokens, labels, token_lengths, mask)

                score, preds = self.model.decode(tokens,token_lengths, mask)
                labels = labels.view(-1).cpu().numpy()
                preds = sum(preds, [])

            else:
                raise NotImplementedError

            # nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            loss.backward()
            self.optimizer.step()

            # preds = preds[labels != config.PAD_VAL]
            org_idxs = np.where(labels != config.PAD_VAL)[0]
            labels = labels[org_idxs]
            # labels = labels[labels != config.PAD_VAL]
            y_true_train.extend(labels.tolist())
            y_pred_train.extend(preds)

            total_loss += loss.item()



        train_loss = total_loss / len(self.train_dataloader)
        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro')
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted')

        # wandb.log({
        #     "train_loss": train_loss,
        #     "train_accuracy": train_accuracy,
        #     "train_f1_score": train_f1,
        #     # "train_precision": train_precision,
        #     # "train_recall": train_recall
        # })



        return train_loss, train_accuracy, train_f1

    def evaluate(self):
        self.model.eval()
        y_true_val = []
        y_pred_val = []
        total_loss = 0

        with torch.no_grad():
            for tokens, labels, token_lengths in tqdm(self.dev_dataloader):
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                if self.classifier == 'crf':
                    mask = (labels != config.PAD_VAL).float()
                    loss = self.model.loss(tokens, labels, token_lengths, mask)

                    score, preds = self.model.decode(tokens,token_lengths, mask)
                    labels = labels.view(-1).cpu().numpy()
                    preds = sum(preds, [])

                elif self.classifier == 'softmax':
                    logits = self.model(tokens, token_lengths)
                    loss = self.loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))
                    preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                    labels = labels.view(-1).cpu().numpy()

                total_loss += loss.item()

                orig_idxs = np.where(labels != config.PAD_VAL)[0]
                labels = labels[orig_idxs]
                y_true_val.extend(labels.tolist())
                y_pred_val.extend(preds.tolist())


        val_loss = total_loss / len(self.dev_dataloader)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted')

        # wandb.log({
        #     "val_loss": val_loss,
        #     "val_accuracy": val_accuracy,
        #     "val_f1_score": val_f1,
        #     # "val_precision": val_precision,
        #     # "val_recall": val_recall
        # })


        return val_loss, val_accuracy, val_f1

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_accuracy, train_f1 = self.train_epoch()
            dev_loss, dev_accuracy, dev_f1 = self.evaluate()
            print(f"Epoch {epoch} train_loss: {train_loss}, train_accuracy: {train_accuracy}, train_F1-score: {train_f1}")
            print(f"Epoch {epoch} val_loss: {dev_loss}, val_accuracy: {dev_accuracy}, val_F1-score: {dev_f1}")

            if dev_loss < self.best_val_loss +0.01 :
                self.best_val_loss = dev_loss
                self.epochs_without_improvement = 0
                # Save the best model if desired
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience or (epoch == 2 and dev_f1 < 0.3):
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {self.best_val_loss}.")
                break


    # def predict(self, batch):
    #     self.model.eval()
    #     y_pred_test = []
    #     with torch.no_grad():
    #         for tokens, labels, token_lengths in tqdm(batch):
    #             tokens, labels = tokens.to(self.device), labels.to(self.device)
    #             logits = self.model(tokens, token_lengths)
    #             preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
    #             y_pred_test.extend(preds.tolist())
    #             torch.cuda.empty_cache()
    #     return y_pred_test

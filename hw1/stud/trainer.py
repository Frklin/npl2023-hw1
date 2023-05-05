import config
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import wandb


class Trainer:
    def __init__(self, model, train_dataloader, dev_dataloader, optimizer, loss_function, device, clip=0, classifier=config.CLASSIFIER):
        '''
        Initializes the Trainer class.

        Args:
            model (nn.Module): The neural network model to be trained.
            train_dataloader (DataLoader): The dataloader for the training data.
            dev_dataloader (DataLoader): The dataloader for the validation data.
            optimizer (optim.Optimizer): The optimizer to be used during training.
            loss_function (callable): The loss function to be used during training.
            device (str): The device to be used during training.
            clip (float): The gradient clipping threshold. Default is 0 (no clipping).
            classifier (str): The classifier type to be used during training. Default is the value specified in config.
        '''

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

        self.clip = clip
        self.classifier = classifier

        # Early stopping
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience = 3
        self.epochs_without_improvement = 0
        self.patience = 10


    def train_epoch(self):
        '''
        Trains the model for one epoch on the training data.

        Returns:
            tuple: A tuple containing the training loss, accuracy, F1 score, precision, and recall.
        '''
        self.model.train()
        total_loss = 0
        y_true_train = []
        y_pred_train = []

        pbar = tqdm(self.train_dataloader, total=20000//config.BATCH_SIZE)

        for tokens, labels, token_lengths, pos, chars in pbar:
            tokens, labels = tokens.to(self.device), labels.to(self.device)

            # Convert POS and character features to tensor and move to device
            pos, chars = pos.to(self.device) if config.POS else None, chars.to(self.device) if config.CHAR else None

            # Convert POS tags to one-hot vectors
            if config.POS:
                pos_vectors = torch.zeros((len(pos), torch.max(token_lengths), config.POS_DIM), dtype=torch.float32).to(self.device)
                
                for i, sentence in enumerate(pos):
                    for j, tag in enumerate(sentence):
                        pos_vectors[i][j] = F.one_hot(tag, num_classes=config.POS_DIM)
            else:
                pos_vectors = None

            # Zero the gradients
            self.optimizer.zero_grad()

            if self.classifier == 'softmax':
                # Forward pass
                logits = self.model(tokens, token_lengths, pos_vectors, chars)

                # Calculate the loss
                loss = self.loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))

                # Backward pass and update weights
                loss.backward()
                self.optimizer.step()

                # Calculate predictions
                preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                labels = labels.view(-1).cpu().numpy()
                org_idxs = np.where(labels != config.PAD_VAL)[0]
                labels = labels[org_idxs]
                preds = preds[org_idxs].tolist()
            
            elif self.classifier == 'crf':
                # CRF Classifier
                m = (labels != config.PAD_VAL)
                mask = m.clone().detach().to(torch.uint8)
                
                self.model.zero_grad()
                loss = self.model.loss(tokens, labels, token_lengths, pos_vectors, chars, mask)
                loss.backward()
                self.optimizer.step()

                preds = self.model.decode(tokens,token_lengths, pos_vectors, chars, mask)
                labels = labels.view(-1).cpu().numpy()
                org_idxs = np.where(labels != config.PAD_VAL)[0]
                labels = labels[org_idxs]
                preds = sum(preds, [])

            else:
                raise NotImplementedError
            
            # Clip gradients
            if self.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            y_true_train.extend(labels.tolist())
            y_pred_train.extend(preds)

            # Update loss
            total_loss += loss.item()

        # Calculate metrics
        train_loss = total_loss / len(self.train_dataloader)
        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro') 
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted')


        return train_loss, train_accuracy, train_f1, train_precision, train_recall

    def evaluate(self):
        '''
        Evaluate the model for one epoch on the training data.

        Returns:
            tuple: A tuple containing the validation loss, accuracy, F1 score, precision, and recall.
        '''
        self.model.eval()
        y_true_val = []
        y_pred_val = []
        total_loss = 0

        with torch.no_grad():
            for tokens, labels, token_lengths, pos, chars in tqdm(self.dev_dataloader):
                tokens, labels = tokens.to(self.device), labels.to(self.device)

                # Convert POS and character features to tensor and move to device
                pos, chars = pos.to(self.device) if config.POS else None, chars.to(self.device) if config.CHAR else None

                # Convert POS tags to one-hot vectors
                if config.POS:
                    pos_vectors = torch.zeros((len(pos), torch.max(token_lengths), config.POS_DIM),dtype=torch.float32).to(self.device)
                    
                    for i, sen in enumerate(pos):
                        for j, tag in enumerate(sen):
                            pos_vectors[i][j] = F.one_hot(tag, num_classes=config.POS_DIM)
                else:
                    pos_vectors = None


                if self.classifier == 'crf':
                    # CRF Classifier
                    m = (labels != config.PAD_VAL)
                    mask = m.clone().detach().to(torch.uint8)
                    loss = self.model.loss(tokens, labels, token_lengths, pos_vectors, chars, mask)
                    labels = self.model(labels, token_lengths, pos_vectors, chars, mask)
                    labels = labels.view(-1).cpu().numpy()
                    org_idxs = np.where(labels != config.PAD_VAL)[0]
                    labels = labels[org_idxs]
                    preds = sum(preds, [])

                elif self.classifier == 'softmax':
                    # Softmax Classifier
                    logits = self.model(tokens, token_lengths, pos_vectors, chars)
                    loss = self.loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))
                    preds = logits.argmax(dim=-1).view(-1).cpu().numpy()
                    labels = labels.view(-1).cpu().numpy()
                    orig_idxs = np.where(labels != config.PAD_VAL)[0]
                    labels = labels[orig_idxs]
                    preds = preds[orig_idxs].tolist()

                # Update loss
                total_loss += loss.item()

                y_true_val.extend(labels.tolist())
                y_pred_val.extend(preds)

        # Calculate metrics
        val_loss = total_loss / len(self.dev_dataloader)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        val_f1 = f1_score(y_true_val, y_pred_val, average='macro')
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted')


        return val_loss, val_accuracy, val_f1, val_precision, val_recall

    def train(self, num_epochs):
        '''
        Train the neural network model for the specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.

        Returns:
            None.
        '''
        best_f1_score = 0  # best F1 score seen so far
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        for epoch in range(num_epochs):
            if epoch >= config.UNFREEZE_EPOCH:
                self.model.unfreeze()
                print("Unfreezing the model")
            
            # Train for one epoch
            train_loss, train_accuracy, train_f1_score, train_precision, train_recall = self.train_epoch()
            
            # Evaluate on the validation set
            val_loss, val_accuracy, val_f1_score, val_precision, val_recall = self.evaluate()
            
            # Log the training and validation metrics
            print(f"Epoch {epoch} train_loss: {train_loss}, train_accuracy: {train_accuracy}, train_F1-score: {train_f1_score}")
            print(f"Epoch {epoch} val_loss: {val_loss}, val_accuracy: {val_accuracy}, val_F1-score: {val_f1_score}")
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_f1_score": train_f1_score,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1_score,
                "val_precision": val_precision,
                "val_recall": val_recall
            })
            
            # Step the learning rate scheduler
            scheduler.step(val_loss)
            
            # Check if the current epoch improves the best validation loss or F1 score
            if val_loss < self.best_val_loss or val_f1_score > best_f1_score:
                best_f1 = val_f1
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), config.MODEL_PATH)
                print(f"Saving model at epoch {epoch+1} with validation loss: {self.best_val_loss}.")
                self.epochs_without_improvement = 0

            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                val_loss = self.best_val_loss
                val_f1 = best_f1
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {self.best_val_loss}.")
                break



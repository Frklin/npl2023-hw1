import sys
sys.path.append('hw1/stud/')
sys.path.append('hw1')

from embeddings import load_embeddings
import torch
import torch.nn as nn
import config
from bilstm import BiLSTM
from load import MyDataset
from utils import seed_everything, collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from trainer import Trainer
from plots import plot_confusion_matrix

import nltk
import wandb
wandb.login()





def main():
    
    # Set seed for reproducibility
    seed_everything(config.SEED)

    # Load embeddings
    embeddings, word2idx = load_embeddings()

    # Define label2idx, a dictionary mapping label names to integer indices
    label2idx = {"O": 0, "B-SENTIMENT": 1, "I-SENTIMENT": 2, "B-CHANGE": 3, "I-CHANGE": 4, "B-ACTION": 5, "I-ACTION": 6, "B-SCENARIO": 7, "I-SCENARIO": 8, "B-POSSESSION": 9, "I-POSSESSION": 10, config.PAD_TOKEN : config.PAD_VAL}
    
    # Define pos2idx, a dictionary mapping POS tags to integer indices
    pos2idx = {x : idx + 1 for idx, x in enumerate(nltk.load('help/tagsets/upenn_tagset.pickle').keys())}
    pos2idx[config.PAD_TOKEN] = config.PAD_IDX
    pos2idx['#'] = len(pos2idx)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = MyDataset(config.TRAIN_PATH, word2idx, label2idx, pos2idx)
    val_dataset = MyDataset(config.VAL_PATH, word2idx, label2idx, pos2idx)
    test_dataset = MyDataset(config.TEST_PATH, word2idx, label2idx, pos2idx)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, shuffle=True)

    # Define model name 
    model_name = get_model_name()

    # Initialize wandb
    # wandb.init(project='nlp_stats',
    #                     name=model_name,
    #                     config={
    #                         "embeddings": config.EMBEDDING_MODEL,
    #                         "model": model_name,
    #                         "classifier": config.CLASSIFIER,
    #                         "hidden_layer": config.HIDDEN_SIZE,
    #                         "POS": config.POS,
    #                         "char": config.CHAR,
    #                         "optimizer": config.OPTIMIZER,
    #                         "batch_size": config.BATCH_SIZE,
    #                         "learning_rate": config.LEARNING_RATE,
    #                         "droprate": config.DROPRATE,
    #                         "weight_decay": config.WEIGHT_DECAY,
    #                         "clip": config.CLIP
    #                         })
    
    # Initialize model
    model = BiLSTM(embeddings, len(config.label2idx), device).to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Initialize loss function
    loss_function = nn.CrossEntropyLoss(ignore_index=config.label2idx[config.PAD_TOKEN])

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, loss_function, device)
    
    # Train model
    # trainer.train(config.EPOCHS)

    # Predict on test set
    true, preds = trainer.predict()

    # Plot confusion matrix
    plot_confusion_matrix(preds, true)












def get_model_name(lstm_layers=config.N_LSTMS, char=config.CHAR, pos=config.POS, classifier=config.CLASSIFIER, hidden_size=config.HIDDEN_SIZE, opt=config.OPTIMIZER, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, dropout=config.DROPRATE, clip=config.CLIP, embeddings=config.EMBEDDING_MODEL):
    lstm_name = ("" if lstm_layers == 1 else "Bi-" if lstm_layers == 2 else "Tri-") + "LSTM"
    if char:
      lstm_name += "-CNN"
    if classifier == "crf":
      lstm_name += "-CRF"
    if pos:
      lstm_name += "-(POS)"
    model_name = '_'.join([embeddings, lstm_name, str(hidden_size)+"HL", opt, str(batch_size)+"BS", str(round(lr,4))+"LR", str(round(dropout,1)) + "DR", str(clip) + "CL"])
    return model_name



def run_epochs(name, train_dataloader, 
               dev_dataloader, 
               embeddings, 
               embeddings_model=config.EMBEDDING_MODEL,
                lstm_layers=config.N_LSTMS, 
                hidden_size=config.HIDDEN_SIZE, 
                classifier=config.CLASSIFIER, 
                opt=config.OPTIMIZER, 
                batch_size=config.BATCH_SIZE, 
                lr=config.LEARNING_RATE, 
                dropout=config.DROPRATE, 
                weight_decay=config.WEIGHT_DECAY, 
                clip=config.CLIP, 
                pos=config.POS,
                char=config.CHAR,
                device=config.DEVICE):
    '''
    Train the model for the specified number of epochs.

    Args:
        name (str): The name of the model.
        train_dataloader (DataLoader): A PyTorch DataLoader object containing the training data.
        dev_dataloader (DataLoader): A PyTorch DataLoader object containing the development data.
        embeddings (Embedding): A PyTorch Embedding object containing the word embeddings.
        embeddings_model (str): The name of the embedding model.
        lstm_layers (int): The number of LSTM layers to use.
        hidden_size (int): The size of the hidden layer.
        classifier (str): The type of classifier to use.
        opt (str): The optimization algorithm to use.
        batch_size (int): The batch size to use.
        lr (float): The learning rate.
        dropout (float): The dropout rate.
        weight_decay (float): The weight decay.
        clip (float): The maximum norm of the gradients.
        pos (bool): Whether or not to use POS tags.
        char (bool): Whether or not to use character-level embeddings.
        device (str): The device to use for training.

    Returns:
        None
    '''
        
    model_name = name + get_model_name(lstm_layers, char, pos, classifier, hidden_size, opt, batch_size, lr, dropout, clip, embeddings_model)
    wandb.init(project='nlp_stats',
                        name="FixedHidden-" + model_name,        
                        config={
                            "embeddings": embeddings_model,
                            "model": model_name,
                            "classifier": classifier,
                            "hidden_layer": hidden_size,
                            "POS": pos,
                            "char": char,
                            "optimizer": opt,
                            "batch_size": batch_size,
                            "learning_rate": lr,
                            "droprate": dropout,
                            "weight_decay": weight_decay,
                            "clip": clip
                            })

    model = BiLSTM(embeddings, len(config.label2idx), device, hidden_size, lstm_layers, dropout, classifier).to(device)
    
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'nadam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)


    loss_function = nn.CrossEntropyLoss(ignore_index=config.label2idx[config.PAD_TOKEN])
    wandb.watch(model, log="all")

    trainer = Trainer(model, train_dataloader, dev_dataloader, optimizer, loss_function, device, clip, classifier)
    trainer.train(config.EPOCHS)
    wandb.finish()






if __name__ == "__main__":
    main()


    
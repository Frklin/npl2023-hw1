# from hw1.stud import *
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
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model import Trainer

# import wandb



def main():
    
    seed_everything(config.SEED)
    
    embeddings, word2idx = load_embeddings()
    
    print("Embeddings shape: ", embeddings.shape)
    print("Word2idx length: ", len(word2idx))

    label2idx = {"O": 0, "B-SENTIMENT": 1, "I-SENTIMENT": 2, "B-CHANGE": 3, "I-CHANGE": 4, "B-ACTION": 5, "I-ACTION": 6, "B-SCENARIO": 7, "I-SCENARIO": 8, "B-POSSESSION": 9, "I-POSSESSION": 10, config.PAD_TOKEN : config.PAD_VAL}
    idx2label = {v: k for k, v in label2idx.items()}

    pos2idx = {config.PAD_TOKEN: config.PAD_IDX, "CC" : 1, "CD" : 2, "DT" : 3, "EX" : 4, "FW" : 5, "IN" : 6, "JJ" : 7, "JJR" : 8, "JJS" : 9, "LS" : 10, "MD" : 11, "NN" : 12, "NNS" : 13, "NNP" : 14, "NNPS" : 15, "PDT" : 16, "POS" : 17, "PRP" : 18, "PRP$" : 19, "RB" : 20, "RBR" : 21, "RBS" : 22, "RP" : 23, "SYM" : 24, "TO" : 25, "UH" : 26, "VB" : 27, "VBD" : 28, "VBG" : 29, "VBN" : 30, "VBP" : 31, "VBZ" : 32, "WDT" : 33, "WP" : 34, "WP$" : 35, "WRB" : 36}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset(config.TRAIN_PATH, word2idx, label2idx, pos2idx)
    val_dataset = MyDataset(config.VAL_PATH, word2idx, label2idx, pos2idx)
    test_dataset = MyDataset(config.TEST_PATH, word2idx, label2idx, pos2idx)

    print("Train dataset length: ", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn)

    model_name = config.EMBEDDING_MODEL + "_" + ("" if config.N_LSTMS == 1 else "Bi-" if config.N_LSTMS == 2 else "Tri-") + "LSTM_" + config.CLASSIFIER + "_" + config.OPTIMIZER + "_" + str(config.LEARNING_RATE) + "LR_" + str(config.DROPRATE) + "DP" 
    print(model_name)
    # wandb.init(
    #   # Set the project where this run will be logged
    #   project="nlp-event-detection",
    #   name=model_name,
    #   # Track hyperparameters and run metadata
    #   config={
    #   "embeddings": "GensimW2V",
    #   "# LSTM": lstm_layers,
    #   "hidden_layer": lstm_units,
    #   "classifier": "Linear",
    #   "learning_rate": lr,
    #   "optimizer": "Adam",
    #   "droprate": dropout,
    #   "batch_size": batch_size
    #   })


    model = BiLSTM(embeddings, len(label2idx), device=device)
    # wandb.watch(model)

    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'nadam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, amsgrad=True)
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    loss_function = nn.CrossEntropyLoss(ignore_index=label2idx[config.PAD_TOKEN])

# model, train_dataloader, dev_dataloader, optimizer, loss_function, device ,clip, classifier = 'softmax'):
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_function, device)
    trainer.train(config.EPOCHS)

    # torch.save(model.state_dict(), 'event_detection_model.pth')






if __name__ == "__main__":
    main()
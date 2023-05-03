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
import wandb
import nltk

wandb.login()




def main():
    
    seed_everything(config.SEED)
    
    embeddings, word2idx = load_embeddings()

    label2idx = {"O": 0, "B-SENTIMENT": 1, "I-SENTIMENT": 2, "B-CHANGE": 3, "I-CHANGE": 4, "B-ACTION": 5, "I-ACTION": 6, "B-SCENARIO": 7, "I-SCENARIO": 8, "B-POSSESSION": 9, "I-POSSESSION": 10, config.PAD_TOKEN : config.PAD_VAL}
    idx2label = {v: k for k, v in label2idx.items()}

    # pos2idx = {config.PAD_TOKEN: config.PAD_IDX, "CC" : 1, "CD" : 2, "DT" : 3, "EX" : 4, "FW" : 5, "IN" : 6, "JJ" : 7, "JJR" : 8, "JJS" : 9, "LS" : 10, "MD" : 11, "NN" : 12, "NNS" : 13, "NNP" : 14, "NNPS" : 15, "PDT" : 16, "POS" : 17, "PRP" : 18, "PRP$" : 19, "RB" : 20, "RBR" : 21, "RBS" : 22, "RP" : 23, "SYM" : 24, "TO" : 25, "UH" : 26, "VB" : 27, "VBD" : 28, "VBG" : 29, "VBN" : 30, "VBP" : 31, "VBZ" : 32, "WDT" : 33, "WP" : 34, "WP$" : 35, "WRB" : 36}
    
    pos2idx = {x : idx + 1 for idx, x in enumerate(nltk.load('help/tagsets/upenn_tagset.pickle').keys())}
    pos2idx[config.PAD_TOKEN] = config.PAD_IDX
    pos2idx['#'] = len(pos2idx)

    char2idx = {config.PAD_TOKEN: config.PAD_VAL, config.UNK_TOKEN: 1, "a" : 1, "b" : 2} 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset(config.TRAIN_PATH, word2idx, label2idx, pos2idx, char2idx)
    val_dataset = MyDataset(config.VAL_PATH, word2idx, label2idx, pos2idx, char2idx)
    test_dataset = MyDataset(config.TEST_PATH, word2idx, label2idx, pos2idx, char2idx)

    print("Train dataset length: ", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,collate_fn=collate_fn, shuffle=True)

    lstm_name = ("" if config.N_LSTMS == 1 else "Bi-" if config.N_LSTMS == 2 else "Tri-") + "LSTM"
    if config.CHAR:
      lstm_name += "-CNN"
    if config.CLASSIFIER == "crf":
      lstm_name += "-CRF"
    if config.POS:
      lstm_name += "-(POS)"
    model_name = '_'.join([config.EMBEDDING_MODEL, lstm_name, str(config.HIDDEN_SIZE)+"HL", config.OPTIMIZER, str(config.BATCH_SIZE)+"BS", str(round(config.LEARNING_RATE,4))+"LR", str(round(config.DROPRATE,1)) + "DR"])
    print(model_name)


    optimizers_settings = {
    'Adam': {'class': optim.Adam, 'params': {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay' :  config.WEIGHT_DECAY}},
    'Nadam': {'class': optim.Adam, 'params': {'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay' : config.WEIGHT_DECAY, 'amsgrad': True}},
    'Adadelta': {'class': optim.Adadelta, 'params': {'rho': 0.9, 'eps': 1e-06, 'weight_decay' : config.WEIGHT_DECAY}},
    'SGD': {'class': optim.SGD, 'params': {'momentum': 0.9, 'weight_decay' : config.WEIGHT_DECAY}}
}
    lrs = [1e-2,1e-3,5e-4,1e-4]
    # for opt_name, opt_setting in optimizers_settings.items():
    #     for lr in lrs:
    #         wandb.init(project='nlp_stats',
    #                     name=model_name,        
    #                   config={
    #                       "embeddings": config.EMBEDDING_MODEL,
    #                       "model": model_name,
    #                       "classifier": config.CLASSIFIER,
    #                       "hidden_layer": config.HIDDEN_SIZE,
    #                       "POS": config.POS,
    #                       "optimizer": opt_name,
    #                       "batch_size": config.BATCH_SIZE,
    #                       "learning_rate": lr,
    #                       "droprate": config.DROPRATE
    #                       })
    #         model = BiLSTM(embeddings, len(label2idx), device=device)
    #         wandb.watch(model, log="all")
    #         optimizer_class = opt_setting['class']
    #         optimizer_params = {**opt_setting['params'], 'lr': lr}
    #         optimizer = optimizer_class(model.parameters(), **optimizer_params)
    #         loss_function = nn.CrossEntropyLoss(ignore_index=label2idx[config.PAD_TOKEN])

    #         trainer = Trainer(model, train_loader, val_loader, optimizer, loss_function, device)
    #         trainer.train(20)
    #         wandb.finish()
            # torch.save(model.state_dict(),  f"{config.SAVE_PATH}/{model_name}.pth")
    model = BiLSTM(embeddings, len(label2idx), device=device)
#     # wandb.watch(model)

    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'nadam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, amsgrad=True)
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    loss_function = nn.CrossEntropyLoss(ignore_index=label2idx[config.PAD_TOKEN])
    wandb.init(project='nlp_stats',
                    name=model_name,        
                    config={
                        "embeddings": config.EMBEDDING_MODEL,
                        "model": model_name,
                        "classifier": config.CLASSIFIER,
                        "hidden_layer": config.HIDDEN_SIZE,
                        "POS": config.POS,
                        "optimizer": config.OPTIMIZER,
                        "batch_size": config.BATCH_SIZE,
                        "learning_rate": config.LEARNING_RATE,
                        "droprate": config.DROPRATE
                        })

    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_function, device)
    trainer.train(config.EPOCHS)

#     torch.save(model.state_dict(), 'event_detection_model.pth')






# def run_epochs(name, label2idx, embeddings_model, lstm_layers, hidden_size, classifier, opt, batch_size, lr, dropout, weight_decay , clip, device):
#     lstm_name = ("" if lstm_layers == 1 else "Bi-" if lstm_layers == 2 else "Tri-") + "LSTM"
#     if config.CHAR:
#       lstm_name += "-CNN"
#     if classifier == "crf":
#       lstm_name += "-CRF"
#     if config.POS:
#       lstm_name += "-(POS)"
#     model_name = '_'.join([name + "-" + embeddings_model, lstm_name, str(hidden_size)+"HL", opt, str(batch_size)+"BS", str(round(lr,4))+"LR", str(round(dropout,1)) + "DR"])
#     # model_name = "EMB"+emb + "_" + lstm_name + "_" + str(lstm_units) + "_" +  "HL_" + "Softmax_" + "Adam_" +str(batch_size)+"BS_" + str(round(lr,4))+"LR_"+ str(round(dropout,1)) + "DR" 
#     wandb.init(project='nlp_stats',
#                         name=model_name,        
#                       config={
#                           "embeddings": embeddings_model,
#                           "model": model_name,
#                           "classifier": classifier,
#                           "hidden_layer": hidden_size,
#                           "POS": config.POS,
#                           "optimizer": opt,
#                           "batch_size": batch_size,
#                           "learning_rate": lr,
#                           "droprate": dropout
#                           })

#     model = BiLSTM(embeddings, len(label2idx), device, hidden_size, lstm_layers, dropout, classifier)
    
#     if opt == 'adam':
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif opt == 'nadam':
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
#     elif opt == 'sgd':
#         optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
#     else:
#         optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)


#     loss_function = nn.CrossEntropyLoss(ignore_index=label2idx['<PAD>'])
#     wandb.watch(model, log="all")

#     trainer = Trainer(model, train_dataloader, dev_dataloader, optimizer, loss_function, device, clip, classifier)
#     trainer.train(num_epochs)
#     wandb.finish()
#     torch.save(model.state_dict(), f"{SAVE_FOLDER}/{round(dev_f1,2)}-{round(dev_loss,2)}-{model_name}.pth")






if __name__ == "__main__":
    main()
import torch
import nltk
nltk.download('tagsets')

# EMBEDDINGS
EMBEDDING_MODEL         = "GenGlove" # "GenW2V", "GenGlove", "Fasttext"
EMBEDDING_SIZE          = 300        # Size of the embedding vector
WINDOW_SIZE             = 5          # Size of the sliding window
NEGATIVE_SAMPLES        = 5          # Negative samples for each word
LABEL_COUNT             = 12         # Number of labels
UNK_TOKEN               = "<UNK>"    # Unknown token
PAD_TOKEN               = "<PAD>"    # Padding token
PAD_IDX                 = 0          # Padding index
PAD_VAL                 = 11         # Padding value
POS                     = True       # Use POS tags
POS_DIM                 = 47         # POS dimension

# CHARACTERS
CHAR                    = True       # Use character embeddings
CHAR_DIM                = 50         # Character embeddings dimension
CHAR_VOCAB_SIZE         = 230        # Character vocabulary size
CNN_FILTERS             = 30        # Number of filters

# LSTM
EPOCHS                  = 30         # Number of epochs
HIDDEN_SIZE             = 2048       # Size of the hidden layer
BATCH_SIZE              = 64         # Size of the batch
N_LSTMS                 = 2          # Number of LSTM layers
LEARNING_RATE           = 1e-4       # Learning rate
DROPRATE                = 0.5        # Dropout rate
CLASSIFIER              = 'softmax'  # 'softmax' or 'crf'
OPTIMIZER               = 'adam'     # 'adam', 'nadam', sgd', 'adagrad'
WEIGHT_DECAY            = 1e-5       # Weight decay
CLIP                    = 1          # Gradient clipping
UNFREEZE_EPOCH          = 3      # Unfreeze embeddings
# OS PATHS
ROOT                    = './data'
TRAIN_PATH              = ROOT + '/train.jsonl'
VAL_PATH                = ROOT + '/dev.jsonl'
TEST_PATH               = ROOT + '/test.jsonl'
MODEL_PATH              = "./model/BiLSTM-CNN-CRF(POS).pt"

EMBEDDINGS_PATH    = "./hw1/stud/EmbModels/" + EMBEDDING_MODEL
SAVE_PATH          = "./model/"

# SEED
SEED                    = 42

DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"


pos2idx = {x : idx + 1 for idx, x in enumerate(nltk.load('help/tagsets/upenn_tagset.pickle').keys())}
pos2idx[PAD_TOKEN] = PAD_IDX
pos2idx['#'] = len(pos2idx)


label2idx = {"O": 0, "B-SENTIMENT": 1, "I-SENTIMENT": 2, "B-CHANGE": 3, "I-CHANGE": 4, "B-ACTION": 5, "I-ACTION": 6, "B-SCENARIO": 7, "I-SCENARIO": 8, "B-POSSESSION": 9, "I-POSSESSION": 10, PAD_TOKEN : PAD_VAL}
idx2label = {v: k for k, v in label2idx.items()}
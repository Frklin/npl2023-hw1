# EMBEDDINGS
EMBEDDING_MODEL         = "GenGlove" # "GenW2V", "GenGlove", "Fasttext"
EMBEDDING_SIZE          = 300        # Size of the embedding vector
WINDOW_SIZE             = 5          # Size of the sliding window
NEGATIVE_SAMPLES        = 5          # Negative samples for each word
UNK_TOKEN               = "<UNK>"    # Unknown token
PAD_TOKEN               = "<PAD>"    # Padding token
PAD_IDX                 = 0          # Padding index
PAD_VAL                 = 11         # Padding value
BOS                     = 12         # BOS index
EOS                     = 13         # EOS index
POS                     = True       # Use POS tags
POS_DIM                 = 38         # POS dimension

# LSTM
EPOCHS                  = 30         # Number of epochs
HIDDEN_SIZE             = 1024       # Size of the hidden layer
BATCH_SIZE              = 16         # Size of the batch
N_LSTMS                 = 2          # Number of LSTM layers
LEARNING_RATE           = 1e-3       # Learning rate
DROPRATE                = 0.5        # Dropout rate
CLASSIFIER              = 'crf'  # 'softmax' or 'crf'
OPTIMIZER               = 'adam'     # 'adam', 'nadam', sgd', 'adagrad'
WEIGHT_DECAY            = 1e-5       # Weight decay

# OS PATHS
ROOT                    = './data'
TRAIN_PATH              = ROOT + '/train.jsonl'
VAL_PATH                = ROOT + '/dev.jsonl'
TEST_PATH               = ROOT + '/test.jsonl'

EMBEDDINGS_PATH    = "./hw1/stud/EmbModels/" + EMBEDDING_MODEL

# SEED
SEED                    = 42








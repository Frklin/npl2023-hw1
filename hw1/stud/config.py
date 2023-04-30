







# EMBEDDINGS
EMBEDDING_MODEL = "GemGlove" # "GenW2V", "GenGlove", "Fasttext"
EMBEDDING_SIZE = 300
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 5
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
PAD_IDX = 0
PAD_VAL = -100


# LSTM
EPOCHS = 25
HIDDEN_SIZE = 1024
BATCH_SIZE = 16
N_LSTMS = 2
LEARNING_RATE = 1e-3
DROPRATE = 0.5
CLASSIFIER = 'softmax'  # 'softmax' or 'crf'
OPTIMIZER = 'adam'  # 'adam', 'nadam', sgd', 'adagrad'
WEIGHT_DECAY = 1e-5


# OS PATHS
ROOT = '../../data'
TRAIN_PATH = ROOT + '/train.jsonl'
VAL_PATH = ROOT + '/dev.jsonl'
TEST_PATH = ROOT + '/test.jsonl'

EMBEDDING_MODEL_PATH = "/EmbModels/" + EMBEDDING_MODEL

# SEED
SEED = 42

import torch

MODEL_TYPE = 'bert-base-uncased'
MAX_TOK_LEN = 64
MODEL_SAVE_FILE = 'models/joint-intent-slot.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Using %s" % (device))

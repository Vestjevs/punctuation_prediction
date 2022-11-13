import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.data import Field, BucketIterator

import random
import time

from model import Encoder, Decoder, Seq2Seq, SEED

# Load data
test_in = list(open('data/test_in_clear.csv', encoding="utf-8"))
test_out = list(open('data/test_out_clear.csv', encoding="utf-8"))

random.seed(SEED)
torch.manual_seed(SEED)


def tokenize(text):
    return [sentence.split() for sentence in text]


SRC = tokenize(test_in)
TRG = tokenize(test_out)

# а я взял - наебал
ind = random.randint(0, len(SRC))
delimiter = int(len(SRC) * 0.6)

train_data, valid_data, test_data = SRC, TRG[0:delimiter], TRG[delimiter:len(SRC)]

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(SRC)
OUTPUT_DIM = len(TRG)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# dont forget to put the model to the right device
model = Seq2Seq(enc, dec, device).to(device)

# Теперь разобраться как осуществить training
import os

import torch
import math
from torch import nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import img

# input = w x h x 3 x seq_len
# CNN => ResNet,VCG, etc.
#       shape = feature_num x seq_len
# Transformer
# output = 1 x class_num

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=2000):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(np.shape(self.pe))
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(0)
        # print(np.shape(self.pe[:, :seq_len]))
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class Transformer(nn.Module):
    def __init__(self, input_size, seq_len, class_num, attn_head, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=attn_head, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.Linear(input_size, class_num)
        self.decoder2 = nn.Linear(seq_len, 1)
        # self.embedding = nn.Embedding(class_num, input_size)
        self.position = PositionalEncoder(input_size)

    def forward(self, input):
        x = self.position(input)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.transpose(x, 1, 2)
        x = self.decoder2(x)
        return x


# input:
# arr = a vector with length equal to the number of classes
# label = label correspond to each class
def read_result(arr, label):
    curmax = 0
    for i in range(len(arr)):
        if arr[i] > arr[curmax]:
            curmax = i
    return label[i], arr[i]


def read_label(path):
    return 0


input_path = "pair/clip11/"

frame_num = 1000
vid_len = 20
class_num = 101
attn_head = 8
dropout = 0.2 # the dropout value

resnet = models.resnet18(pretrained=True)
model = Transformer(input_size=frame_num, seq_len=vid_len, class_num=class_num, attn_head=attn_head, dropout=dropout)

x = img.import_images2(resnet, input_path)
print(np.shape(x))
x = model(x)
print(np.shape(x))
print(x)


def train(input_path, epoch=100):
    resnet = models.resnet18(pretrained=True)
    model = Transformer(input_size=frame_num, seq_len=vid_len, class_num=class_num, attn_head=attn_head, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for e in range(epoch):
        # import data
        folders = os.listdir(input_path)
        for folder in folders:
            folder_path = input_path + "/" + folder + "/"
            optimizer.zero_grad()
            x = img.import_images2(resnet, folder_path)
            target = read_label(folder_path)
            x = model(x)
            print(x)

            loss = criterion(x, target)
            loss.backward()
            optimizer.step()

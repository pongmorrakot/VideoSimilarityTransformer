import os
import random
import torch
import math
from torch import nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import statistics
from copy import deepcopy

import img
from train_prep import prep

# input = w x h x 3 x seq_len
# CNN => ResNet,VCG, etc.
#       shape = feature_num x seq_len
# Transformer
# output = 1 x class_num

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# dev = "cpu"
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
        # print(np.shape(x))
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

def read_annotation(input_path):
    return open(input_path, "r").readlines()

# used for testing
# compare the output of the network with the label
# TODO: implement this using eval()
def read_label(label, output):
    label = int(label)
    class_index, confidence = eval(output)
    print(str(label) + "\t" + str(class_index) + ":" + str(confidence))
    if label == class_index:
        return True
    else:
        return False


def eval(output):
    sc = nn.Softmax(dim=0)
    output = sc(output).cpu().detach().numpy()
    max_index = np.argmax(output) + 1 # +1 as the class index of UCF101 dataset starts at 1, and not 0
    return max_index, output[max_index]


weight_path = "classifier.weight"
feature_num = 1000
vid_len = 200
class_num = 101
attn_head = 4
dropout = 0.2 # the dropout value
learning_rate = 0.001

# resnet = models.resnet18(pretrained=True)
# model = Transformer(input_size=frame_num, seq_len=vid_len, class_num=class_num, attn_head=attn_head, dropout=dropout)
#
# x = img.import_images2(resnet, input_path)
# print(np.shape(x))
# x = model(x)
# print(np.shape(x))
# print(x)


def train(input_path, epoch=30):
    print("Load Model")
    print("Device:\t" + str(device))
    resnet = models.resnet18(pretrained=True).to(device)
    model = Transformer(input_size=feature_num, seq_len=vid_len, class_num=class_num, attn_head=attn_head, dropout=dropout).to(device)
    if os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))
    print("Initialize Training Function")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    data_list = read_annotation(input_path)
    print("Training Start")
    m = nn.Softmax(dim=0)
    for e in range(epoch):
        random.shuffle(data_list)
        for b in range(len(data_list)):
            label, folder_path = data_list[b].split()
            x = img.import_images2(resnet, folder_path).to(device)
            optimizer.zero_grad()
            # print(np.shape(x))
            # target = read_label(int(label), class_num).to(device)
            target = torch.tensor([int(label)-1]).to(device)
            x = model(x).squeeze(2)
            # print("output: " + str(np.shape(x)))
            # print(target)
            loss = criterion(x, target)
            print("Epoch: " + str(e) + "/" + str(epoch) + "\t" + str(b) + "/" + str(len(data_list)) + "\t" + folder_path[len(folder_path)-30:] + "\t" + str(loss))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), weight_path)
        print("Model Saved")


# TODO: split the training set into k sets
# pick a set as the test set, the rest as train set
# for each epoch: if the loss on the test set starts to increase, stop training
def kfold_train(k, input_path):
    print("Load Model")
    print("Device:\t" + str(device))
    resnet = models.resnet18(pretrained=True).to(device)
    model = Transformer(input_size=feature_num, seq_len=vid_len, class_num=class_num, attn_head=attn_head, dropout=dropout).to(device)
    if os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))
    print("Initialize Training Function")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Process Data")
    data_list = read_annotation(input_path)
    fold_size = len(data_list) / k
    folds = []
    random.shuffle(data_list)
    # create fold
    for i in range(k):
        fold = []
        for j in range(fold_size):
            if data_list:
                fold.append(data_list.pop())
        folds.append(fold)
    # train
    print("Start Training")
    for i in range(k):
        # pick test/train set
        testlist = folds[i]
        trainlist = []
        for j in range(k):
            if j != i:
                trainlist += folds[j]
        random.shuffle(trainlist)
        done = False
        prev_loss = math.inf
        e = 0
        while not done:
            train_loss = torch.zeros(1)
            for b in range(len(trainlist)):
                label, folder_path = trainlist[b].split()
                x = img.import_images2(resnet, folder_path).to(device)
                optimizer.zero_grad()
                # print(np.shape(x))
                # target = read_label(int(label), class_num).to(device)
                target = torch.tensor([int(label)-1]).to(device)
                x = model(x).squeeze(2)
                # print("output: " + str(np.shape(x)))
                # print(target)
                loss = criterion(x, target)
                train_loss += loss
                # print("Epoch: " + str(e) + "\t" + str(b) + "/" + str(len(trainlist)) + "\t" + folder_path[len(folder_path)-30:] + "\t" + str(loss))
                loss.backward()
                optimizer.step()

            # validation
            cur_loss = torch.zeros(1)
            with torch.no_grad():
                for b in range(len(testlist)):
                    label, folder_path = testlist[b].split()
                    x = img.import_images2(resnet, folder_path).to(device)
                    target = torch.tensor([int(label)-1]).to(device)
                    x = model(x).squeeze(2)
                    cur_loss += criterion(x, target)

            print("Fold: " + str(i) + "\t" + "Epoch: " + str(e) + "\t" + str(train_loss) + "Test Loss:\t" + str(cur_loss))

            if prev_loss < cur_loss:
                done = True
            else:
                torch.save(model.state_dict(), weight_path)
            e += 1







def test(input_path):
    print("Load Model")
    print("Device:\t" + str(device))
    resnet = models.resnet18(pretrained=True).to(device)
    model = Transformer(input_size=feature_num, seq_len=vid_len, class_num=class_num, attn_head=attn_head, dropout=dropout).to(device)
    if os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))
    print("Initialize Test Function")
    data_list = read_annotation(input_path)
    # random.shuffle(data_list)
    correct = 0
    total = 0
    for entry in data_list:
        label, folder_path = entry.split()
        print(folder_path)
        x = img.import_images2(resnet, folder_path).to(device)
        x = model(x).squeeze()
        if read_label(label, x):
            correct += 1
        total += 1
        print(str(correct) + "/" + str(total))
    return correct/total


# TODO: use 3-fold accuracy (Top-1)
# the average 3-fold cross validation accuracy
def x_validate(trainlist, testlist, epoch=30):
    print("Cross Validation")
    if os.path.isfile(weight_path):
        os.remove(weight_path)
    print("Weight deleted")
    score = []
    print("Data Preparation")
    prep(trainlist, "train.txt")
    prep(testlist, "test.txt")
    print("Training")
    train("train.txt", epoch)
    print("Testing")
    score.append(test("test.txt"))
    print(score)
    print(statistics.mean(score))

x_validate("ucfTrainTestlist/trainlist01.txt", "ucfTrainTestlist/testlist01.txt")

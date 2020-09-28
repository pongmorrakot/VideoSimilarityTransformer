import numpy
import torch
import torchvision
from torch import nn
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import img

# the input matrix would be the size of img_size x num_frame

# img_size = 2400
# num_frame = 30
# query = Input x query_weight
# key = Input x key_weight
# value = input x value_weight
# self-attention = softmax((q x transpose(k))/sqrt(dk)) * v
# dk = dimension of k(key)

# TODO:
# 1. functions to turn video into matrix
# 2. siamese model


# class SiameseTransformer(nn.Module):
#     def __init__(self):
#         return 0
#
#     def forward(self):
#         return 0
#
#
# def contrastive_loss():
#     return 0

learning_rate = 0.001

weight_path = "transformer.weight"

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# dev = "cpu"
device = torch.device(dev)


class siameseTransformer(nn.Module):
    def __init__(self, pixel_num=1000, feature_num=1024, frame_num=20):
        super().__init__()
        self.pixel_num = pixel_num
        self.feature_num = feature_num
        self.frame_num = frame_num
        self.top = nn.Linear(pixel_num, feature_num)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_num, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.bottom = nn.Linear(feature_num, 512)
        # self.bottom2 = nn.Linear(64, frame_num)

    def forward(self, in_tensor):
        # x = in_tensor
        # in_tensor = torch.transpose(in_tensor, 0, 1)
        # x = torch.empty((1, 2000, self.pixel_num)).to(device)
        x = in_tensor
        # print(np.shape(x))
        x = self.top(x)
        # x = torch.transpose(x, 1, 2)
        # print(np.shape(x))
        # print(x)
        x = self.encoder(x)
        # x = torch.transpose(x, 1, 2)
        # print(np.shape(x))
        # print(x)
        x = self.bottom(x)
        # x = self.bottom2(x)
        return x[0]

    def twin_forward(self, input1, input2):
        ret1 = self.forward(input1)
        ret2 = self.forward(input2)
        # torch.cat((ret1, ret2),)
        return ret1, ret2

    # implement how to better quantify the similarity
    def eval(self, output1, output2):
        # pdist = nn.PairwiseDistance()
        # euclidean_distance = pdist(output1, output2)

        x, y = self.twin_forward(output1, output2)
        print(np.shape(x))
        print(x)
        print(y)
        # label = 1
        # margin = 2.0

        euclidean_distance = torch.cdist(x, y)
        score = torch.mean(euclidean_distance, 1)
        print(np.shape(score))
        print(score)
        max_score = torch.max(score)
        print(max_score)
        return max_score
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        # print("distance")
        # print(euclidean_distance)
        # print(loss_contrastive)
        # return euclidean_distance


# def loss(x, y):
#     # x, y are 2D matrices each representing an image sequence
#     z = torch.abs(x - y)
#     # print(z)
#     z = torch.sum(z, dim=1)
#     # print(z)
#     # z = torch.sigmoid(z)
#     # print(z)
#     z = torch.mean(z)
#     # print(z)
#     return z
#
#
# def score(x, y):
#     # x, y are 2D matrices each representing an image sequence
#     arr = []
#     for i in range(np.shape(x)[0]):
#         x = torch.roll(x, 1, 0)
#         val = loss(x, y)
#         arr.append(val)
#         # print(x)
#         # print(y)
#         # print(val)
#         # print(i)
#     # print(arr)
#     return min(arr)

class SimilarityLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(SimilarityLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # pdist = nn.PairwiseDistance()
        euclidean_distance = torch.cdist(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# batch
# 50 epoch seems to be a good number; 100 epoch is too much
def train(path, label, epoch=50):
    print("load model")
    model = siameseTransformer(frame_num=2000).to(device)
    if os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))
    print("loss function")
    lossfunc = SimilarityLoss().to(device)
    print("import image")
    batch = img.import_all(path)
    # resnet = torchvision.models.resnet18(pretrained=True)
    # arr = img.import_images(resnet, "4_retake/")
    # batch = [[arr, arr]]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for t in range(epoch):
        i = 0
        for pair in batch:
            s1 = pair[0]
            s2 = pair[1]
            print("Epoch: " + str(t) + "\tPair: " + str(i))
            x, y = model.twin_forward(s1, s2)
            loss = lossfunc(x, y, label)
            # print(x)
            # print(y)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            # if loss < best_loss:
            #     best_loss = loss
            #     # save the model
            #     torch.save(model.state_dict(), weight_path)
            #     print("Weight saved")
        torch.save(model.state_dict(), weight_path)
        print("Weight saved")


def test():
    print("start")
    model = siameseTransformer(frame_num=2000).to(device)
    print("load model")
    if os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))
    lossfunc = SimilarityLoss().to(device)
    print("import images")
    pair = img.import_pair("pair/")
    print("eval")
    model.eval(pair[0], pair[1])

# pos = os.listdir("vcdb_positive/")
# neg = os.listdir("vcdb_negative/")
# while pos or neg:
#     if pos:
#         train("vcdb_positive/" + pos.pop() + "/", label=1)
#     if neg:
#         train("vcdb_negative/" + neg.pop() + "/", label=0)


test()

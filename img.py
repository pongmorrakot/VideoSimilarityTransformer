import math
import os
import numpy
import torch
from PIL import Image
from torch import nn
import torchvision.models as models
from torchvision import transforms

debug = True

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# dev = "cpu"
device = torch.device(dev)

preprocess = transforms.Compose([
    transforms.Resize((400, 225)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def import_images2(model, path):
    images = os.listdir(path)
    length = len(images)
    arr = torch.zeros((len(images), 3, 400, 225)).to(device)
    i = 0
    for img in images:
        input_image = Image.open(path + img)
        input_tensor = preprocess(input_image).to(device)
        arr[i] = input_tensor
        i += 1
    with torch.no_grad():
        arr = model(arr)
    return arr


def import_images(model, path):
    # print(device)
    images = os.listdir(path)
    length = len(images)
    array = torch.zeros((1, 2000, 1000)).to(device)
    i = 0
    print("# frames:\t" + str(len(images)))
    for img in images:
        # resize image tp 400x225
        # arr = torch.from_numpy(numpy.array(Image.open(path + img).resize((400, 225))))
        input_image = Image.open(path + img)
        input_tensor = preprocess(input_image).to(device)
        # arr = model(input_tensor)
        with torch.no_grad():
            arr = model(input_tensor)
        # print(arr)
        array[0][i] = arr
        i += 1
    if debug:
        print(array)
        print(path + "\tDone")
    return array


# import_images()
def import_all(folder_path):
    resnet = models.resnet50(pretrained=True).to(device)
    array = []
    # folder_path = "vcdb/"
    folders = os.listdir(folder_path)
    lengths = []
    i = 0
    for f in folders:
        print("Import:\t" + str(i) + "/" + str(len(folders)))
        vid_path = folder_path + f + "/"
        clips = os.listdir(vid_path)
        pair = []
        for c in clips:
            pair.append(import_images(resnet, vid_path + c + "/"))
        array.append(pair)
        i += 1
    return array
# print("max length:" + str(max(lengths)))


def import_pair(path):
    resnet = models.resnet50(pretrained=True).to(device)
    folders = os.listdir(path)
    pair = []
    for f in folders:
        vid_path = path + f + "/"
        pair.append(import_images(resnet, vid_path))
    return pair

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
import json
from PIL import Image
import argparse

'''
    Prediction Script
    To be run and followed after successfully saving a checkpoint to 'checkpoint.pth'
    from Train.py

    Required Arguments :
        1) image_path : e.g. './flowers/test/1/image_06743.jpg'
        2) path (aka checkpoint) : e.g. 'checkpoint.pth'

    Contents:

'''
import utils
import class_functions

def main():

    args = get_arguments()
    print("Arguments from command : ", args)

    image_path = args.image_path
    path = args.checkpoint
    topk = args.topk
    category_names = args.category_names
    device = 'gpu' if args.gpu else 'cpu'

    train_loader, valid_loader, test_loader, train_data = utils.load_data()

    print("Loading Checkpoint...")

    model, checkpoint = class_functions.load_checkpoint(path)

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = { val: key for key, val in class_to_idx.items() }

    cat_to_name = class_functions.cat_to_name(category_names)

    probabilities, classes = class_functions.predict(image_path, model,
                                                     idx_to_class, device, topk)

    labels = [cat_to_name[x] for x in classes]

    print("Classes Predicted... : {}".format(len(classes)))
    print("Classes... :", labels)
    print("Probability of Predictions... : ",probabilities)
    print("Classifying is finished...")


def get_arguments():

    parser = argparse.ArugmentParser(description='Predict and Classify an image.')
    parser.add_argument('image_path', default='./flowers/test/102/image_08030.jpg', nargs='*', action="store", type = str)
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
    parser.add_argument('--topk', dest="topk", default=5, action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', dest="gpu", action="store", default='cpu', type=str, help="Use GPU? type 'gpu' or 'cpu'")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
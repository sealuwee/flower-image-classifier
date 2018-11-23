import numpy as np
import torch
import torchvision
import argparse
import json
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict

'''
Class Functions
import to train.py and predict.py

Required import of utils.py to load_data and process_image

Contents:

setup_model(): set up before training
train_model(): trains the model given arch
test_check(): checks accuracy of the model with the test set of images
save_checkpoint(): save checkpoint to path == 'checkpoint.pth'
load_checkpoint(): loads saved checkpoint from path == 'checkpoint.pth'
predict(): prediction function
'''

# Global Variables

import utils

train_data, train_loader, valid_loader, test_loader = utils.load_data()

archs = {
    'alexnet' : 9216,
    'vgg' : 25088
}

# Functions

def setup_model(arch, hidden_layer, output_size):

    if arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'vgg':
        model = models.vgg16(pretrained = True)
    else:
        print("{} is not a valid model for training. Please choose from the following: 'alexnet' or 'vgg'".format(arch))

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('hidden1',nn.Linear(archs[arch], hidden_layer)),
                                ('relu1', nn.ReLU()),
                                ('dropout', nn.Dropout(p=0.5)),
                                ('hidden2',nn.Linear(hidden_layer, output_size)),
                                ('relu2', nn.ReLU()),
                                ('output',nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    learning_rate = 0.001
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_model(model, train_loader, valid_loader, epochs, criterion, optimizer, device):

    print("******** Training Model ********")

    epochs = epochs
    print_every = 40
    steps = 0

    model.train()

    device = torch.device('cuda:0' if device=='gpu' and torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):

        running_loss = 0

        for i, data in enumerate(train_loader):

            steps += 1

            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            running_loss = 0

            if steps % print_every == 0:

                valid_loss, accuracy = validation(model, valid_loader, criterion)

                print("Epoch: {}/{} ".format(epoch + 1, epochs),
                      "\n Training Loss: {:.4f} ".format(running_loss/print_every),
                      "\n Validation Loss: {:.4f}".format(valid_loss/len(valid_loader)),
                      "\n Accuracy: {:.4f}".format(accuracy/len(valid_loader)))

            print("******** Training is Finished ********",
                  "\nModel Requirements Loading...",
                  "\n******** Epochs: {} ********".format(epochs),
                  "\n******** Steps: {} ********".format(steps))

def validation(model, valid_loader, criterion):

    accuracy = 0
    valid_loss = 0

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model.to(device)

    model.eval()

    with torch.no_grad():

        for images, labels in valid_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()

    model.train()

    return valid_loss, accuracy

def test_check(model, test_loader):

    # Help from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    correct = 0
    total = 0
    test_data_steps = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

           print("Checking Accuracy on {} images in the Test Set".format(test_data_steps),
                 "\n Accuracy : %d %% " % (100 * correct / total))

def load_checkpoint(path):
    '''
    Function that loads a checkpoint and rebuilds the model
    '''
    checkpoint = torch.load(path)

    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint

def predict(image_path, model, idx_to_class, topk=5, device='cpu'):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    input_process = utils.process_image(image_path).squeeze()

    model.eval()

    if device == 'gpu' and torch.cuda.is_available():

        model = model.cuda()

        with torch.no_grad():
            if device == 'gpu':
                output = model(torch.from_numpy(input_process).float().cuda().unsqueeze_(0))
            else:
                output = model(torch.from_numpy(input_process).float().cpu().unsqueeze_(0))

    probabilities = F.softmax(output.data, dim=1)

    top_idx = torch.topk(probabilities, topk)

    ps = top_idx[0][0].cpu().numpy()
    classes = [idx_to_class[x] for x in top_idx[1][0].cpu().numpy()]

    return ps, classes

def cat_to_name(jsonfile):

    with open(jsonfile, 'r') as f:

        cat_to_name = json.load(f)

    return cat_to_name
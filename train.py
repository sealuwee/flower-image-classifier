import torch
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

'''
    Training Script
    import utils.py and class_functions.py to run correctly
    Calls main() function to run
    Arguments are parsed in the beginning to create variables for convenience
    e.g. for checkpoint and arguments for functions from class_functions.py

    Run train.py and continue with predict.py

    Contents:

    main(): train.py script

    get_arguments():

        returns:
        args = parser.parse_args() for simplicity of argument parser

'''

import utils
import class_functions

archs = {
    'alexnet' : 9216,
    'vgg' : 25088
}

def main():

    args = get_arguments()
    print("Arguments from command : ", args)

    data_dir = './flowers'
    arch = args.arch
    path = args.save_dir
    device = 'gpu' if args.gpu else 'cpu'
    hidden_layer = args.hidden_units
    output_size = args.output_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    print("Initializing 'Train.py'...")

    train_loader, valid_loader, test_loader, train_data = utils.load_data()

    model, optimizer, criterion = class_functions.setup_model(arch, hidden_layer, output_size)

    class_functions.train_model(model, train_loader, valid_loader, epochs, criterion, optimizer, device)

    class_functions.test_check(model, test_loader)

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
                'hidden_layer': hidden_layer,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'input_size': archs[arch],
                'output_size': 102,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'optimizer':optimizer,
                'classifier': model.classifier}

    torch.save(checkpoint, path)

    print("Model Successfully Saved...")

def get_arguments():

    parser = argparse.ArgumentParser(description='Training Flower Image Classifier')
    parser.add_argument('--gpu', dest="gpu", action="store", default='cpu', type=str, help="Use GPU? type 'gpu' or 'cpu'")
    parser.add_argument('--save_dir', dest="save_dir",action="store", default="checkpoint.pth", type=str, help="Path of Saved Model")
    parser.add_argument('--learning_rate', dest="learning_rate",action="store", type=float, default=0.001, help="Learning Rate... Default Value = 0.001")
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5, help="Number of Epochs... Default Value = 10")
    parser.add_argument('--arch', dest="arch", action="store", default="alexnet", type = str, help="Model Architecture... [alexnet, vgg16]... Default = 'alexnet'")
    parser.add_argument('--hidden_units', dest="hidden_units", action="store", default=100, type=int, help="Units in Hidden Layer... Default Value = 100")
    parser.add_argument('--output_size', dest="output_size", action="store", default=102, type=int, help="Output Size, Recommended: Do not Change")
    parser.add_argument('data_dir', action="store", default="./flowers", type=str, help="Finding Data Directory")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
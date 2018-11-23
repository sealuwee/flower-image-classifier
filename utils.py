import numpy as np
import torch
import matplotlib as plt
from torchvision import datasets, transforms, models, utils
from PIL import Image
from torch.utils.data import DataLoader

'''
    Functional Utility
    import to class_functions.py, train.py and predict.py

    Contents:

    load_data(data_dir) : loads data, e.g. train_data, valid_loader, ...
    process_image(image) : processes image given its path, returns a tensor
'''

batch_size = 64

def load_data():

    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose(
        [transforms.RandomRotation(30),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    validation_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    testing_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(validate_data, batch_size=batch_size)

    return train_data, train_loader, valid_loader, test_loader

def process_image(image_path):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an array
    '''
    # https://pillow.readthedocs.io/en/5.3.x/reference/Image.html
    # Pillow image coordinate system
    # https://pillow.readthedocs.io/en/5.3.x/handbook/concepts.html#coordinate-system

    print("Processing Image from ... {}".format(image_path))

    pil_img = Image.open(image_path)

    size = 224
    width, height = pil_img.size

    x = (width - size) / 2
    y = (height - size) / 2

    cropped_img = pil_img.crop((x, y, (x+size), (y+size)))
    np_img_array = np.array(cropped_img)
    np_img_array = np_img_array/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_tensor = (np_img_array - mean) / std
    img_tensor = np.transpose(np_img_array,(2,0,1))

    return img_tensor
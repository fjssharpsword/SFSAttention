from pathlib import Path
import torchvision.datasets as dset
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from torchvision import transforms, datasets
from data.CXR import get_box_dataloader_VIN
from data.Fundus import get_train_dataset_fundus, get_test_dataset_fundus
from data.Corpus import get_train_dataset_corpus
n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x

def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()

def one_hot_encode(target):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    num_classes = 10
    one_hot_encoding = F.one_hot(torch.tensor(target),num_classes)

    one_hot_encoding = torch.as_tensor(one_hot_encoding, dtype=torch.float32) #turn long to float

    return one_hot_encoding


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset

def get_CXR():
    image_shape = (112, 112, 3)
    num_classes = 15
    train_dataset = get_box_dataloader_VIN(is_train = True)
    test_dataset = get_box_dataloader_VIN(is_train = False)
    return image_shape, num_classes, train_dataset, test_dataset


def get_Fundus():
    image_shape = (112, 112, 3)
    num_classes = 5
    train_dataset = get_train_dataset_CXR()
    test_dataset = get_test_dataset_CXR()
    return image_shape, num_classes, train_dataset, test_dataset

def get_corpus():
    image_shape = (80, 192, 1)
    num_classes = 20
    train_dataset = get_train_dataset_corpus()
    test_dataset = get_train_dataset_corpus()
    return image_shape, num_classes, train_dataset, test_dataset

"""
def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset
"""

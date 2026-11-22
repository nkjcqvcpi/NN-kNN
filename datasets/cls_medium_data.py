import torch
from torchvision import datasets, transforms
import os

from .paths import datasets_root

# DATA_ROOT overrides the shared NN-kNN data directory (datasets.paths); nothing
# is created at import time.
DATA_ROOT = os.environ.get("DATA_ROOT") or None


def _root(root, *sub):
    if root:
        return root
    return os.path.join(DATA_ROOT or str(datasets_root()), *sub)


def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=2)
    mean, std, total_samples = 0.0, 0.0, 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(1).sum(0)
        std += images.std(1).sum(0)
        total_samples += batch_samples
    mean /= total_samples
    std /= total_samples
    return mean.item(), std.item()


def MNIST(root=None):
    root = _root(root)
    initial_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=initial_transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"MNIST mean={mean:.4f}, std={std:.4f}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    return X_train, y_train, X_test, y_test


def CIFAR10(root=None):
    root = _root(root)
    initial_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=initial_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"CIFAR-10 mean={mean}, std={std}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    return X_train, y_train, X_test, y_test

def CIFAR100(root=None):
    root = _root(root)
    initial_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=initial_transform)
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"CIFAR-100 mean={mean}, std={std}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    return X_train, y_train, X_test, y_test


def SVHN(root=None):
    root = _root(root, "svhn")
    initial_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.SVHN(root=root, split="train", download=True, transform=initial_transform)
    test_dataset = datasets.SVHN(root=root, split="test", download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"SVHN mean={mean}, std={std}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = datasets.SVHN(root=root, split="train", transform=transform)
    test_dataset = datasets.SVHN(root=root, split="test", transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    return X_train, y_train, X_test, y_test


# Placeholder datasets
def ImageNet32(root=None): pass
def TinyImageNet(root=None): pass
def CoLA(root=None): pass
def AGNews(root=None): pass

def SST1():
    from datasets import load_dataset
    dataset = load_dataset("sst", "default")
    def _map_to_classes(example):
        if example['label'] < 0.2: example['label'] = 0
        elif example['label'] < 0.4: example['label'] = 1
        elif example['label'] < 0.6: example['label'] = 2
        elif example['label'] < 0.8: example['label'] = 3
        else: example['label'] = 4
        return example
    dataset = dataset.map(_map_to_classes)
    return dataset['train'], dataset['validation'], dataset['test']

def SST2():
    from datasets import load_dataset
    dataset = load_dataset("sst", "default")
    dataset = dataset.filter(lambda ex: ex['label'] < 0.4 or ex['label'] >= 0.6)
    def _map_to_binary_classes(example):
        if example['label'] < 0.4: example['label'] = 0
        else: example['label'] = 1
        return example
    dataset = dataset.map(_map_to_binary_classes)
    return dataset['train'], dataset['validation'], dataset['test']


DATATYPES = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "imagenet32": ImageNet32,
    "tinyimagenet": TinyImageNet,
    "sst1": SST1,
    "sst2": SST2,
    "cola": CoLA,
    "agnews": AGNews,
}

def Cls_medium_data(dataset: str):
    return DATATYPES[dataset]()

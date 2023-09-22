from typing import Any
import torch
from torch import nn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class CifarDataset(Dataset):
    def __init__(
        self, path="./data", train=True, transforms=None, num_classes=10
    ) -> None:
        super().__init__()
        self.data = CIFAR10(path, download=True, train=train, transform=transforms)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        image = self.data[index][0]
        label = self.data[index][1]
        label_onehot = nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=self.num_classes
        )

        return image, label_onehot


class MNISTDataset(Dataset):
    def __init__(
        self, path="./data", train=True, transforms=None, num_classes=10
    ) -> None:
        super().__init__()
        self.data = MNIST(path, download=True, train=train, transform=transforms)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        image = self.data[index][0]
        label = self.data[index][1]
        label_onehot = nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=self.num_classes
        )

        return image, label_onehot


class CifarDataModule(LightningDataModule):
    def __init__(
        self, path="./data", transforms=None, num_classes=10, batch_size=32
    ) -> None:
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.num_classes = num_classes
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        self.train_dataset = CifarDataset(
            path=self.path,
            train=True,
            transforms=self.transforms,
            num_classes=self.num_classes,
        )
        self.val_dataset = CifarDataset(
            path=self.path,
            train=False,
            transforms=self.transforms,
            num_classes=self.num_classes,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


class MNISTDataModule(LightningDataModule):
    def __init__(
        self, path="./data", transforms=None, num_classes=10, batch_size=32
    ) -> None:
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.num_classes = num_classes
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        self.train_dataset = MNISTDataset(
            path=self.path,
            train=True,
            transforms=self.transforms,
            num_classes=self.num_classes,
        )
        self.val_dataset = MNISTDataset(
            path=self.path,
            train=False,
            transforms=self.transforms,
            num_classes=self.num_classes,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

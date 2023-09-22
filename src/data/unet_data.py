from typing import Any
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class OxfordPetDataset(Dataset):
    def __init__(
        self, path="./data", split="trainval", transforms=None, mask_transforms=None
    ) -> None:
        super().__init__()
        self.data = OxfordIIITPet(root=path, split=split, target_types="segmentation")
        self.transforms = transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        image = self.data[index][0]
        mask = self.data[index][1]

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return image, mask


class OxfordPetDataModule(LightningDataModule):
    def __init__(
        self, path="./data", transforms=None, mask_transforms=None, batch_size=32
    ) -> None:
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        self.train_dataset = OxfordPetDataset(
            path=self.path,
            split="trainval",
            transforms=self.transforms,
            mask_transforms=self.mask_transforms,
        )
        self.val_dataset = OxfordPetDataset(
            path=self.path,
            split="test",
            transforms=self.transforms,
            mask_transforms=self.mask_transforms,
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

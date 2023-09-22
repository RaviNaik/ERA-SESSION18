import torch
from torchvision import transforms

image_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((256, 256))]
)

mask_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: (x - 1).squeeze().type(torch.LongTensor)),
    ]
)

import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F


def plot_vae_samples(datamodule, class_map, num_images=5):
    fig, ax = plt.subplots(1, num_images, figsize=(15, 10))
    for i in range(num_images):
        x = datamodule.val_dataset[i][0]
        label = datamodule.val_dataset[i][1].argmax().item()

        ax[i].imshow(x.permute(1, 2, 0))
        ax[i].set_title(f"Label-{class_map[label]}")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.show()


def plot_vae_results(model, datamodule, class_map, num_images=5, num_classes=10):
    fig, ax = plt.subplots(num_images, 2, figsize=(8, 30))
    # fig.subplots_adjust(wspace=0, hspace=0)

    for i in range(num_images):
        image = datamodule.val_dataset[i][0]
        actual_label_onehot = datamodule.val_dataset[i][1]
        actual_label = actual_label_onehot.argmax().item()

        class_list = list(range(num_classes))
        class_list.remove(actual_label)
        wrong_label = random.choice(class_list)
        wrong_label_onehot = F.one_hot(
            torch.tensor(wrong_label, dtype=torch.long), num_classes=10
        )

        predicted, _, _, _ = model(image.unsqueeze(0), wrong_label_onehot.unsqueeze(0))

        ax[i][0].imshow(image.permute(1, 2, 0).numpy())
        ax[i][0].set_title(f"Actual Label: {class_map[actual_label]}")

        ax[i][1].imshow(predicted.detach().squeeze(0).permute(1, 2, 0).numpy())
        ax[i][1].set_title(f"Wrong Label: {class_map[wrong_label]}")

        for a in ax[i]:
            a.set_xticklabels([])
            a.set_yticklabels([])
    plt.tight_layout()
    plt.show()


def plot_unet_samples(datamodule, num_images=5):
    fig, ax = plt.subplots(num_images, 2, figsize=(10, 15))
    for i in range(num_images):
        x = datamodule.val_dataset[i][0]
        target = datamodule.val_dataset[i][1]

        ax[i][0].imshow(x.permute(1, 2, 0))
        ax[i][0].set_title(f"Input Image-{i+1}")

        ax[i][1].imshow(target.permute(1, 2, 0))
        ax[i][1].set_title(f"Target Image-{i+1}")

        for a in ax[i]:
            a.set_xticklabels([])
            a.set_yticklabels([])
    plt.show()


def plot_unet_results(model, datamodule, num_images=5):
    fig, ax = plt.subplots(num_images, 3, figsize=(15, 15))
    # fig.subplots_adjust(wspace=0, hspace=0)

    for i in range(num_images):
        x = datamodule.val_dataset[i][0]
        target = datamodule.val_dataset[i][1]
        predicted = model(x.unsqueeze(0))
        ax[i][0].imshow(x.permute(1, 2, 0))
        ax[i][0].set_title(f"Input Image-{i+1}")

        ax[i][1].imshow(target.permute(1, 2, 0))
        ax[i][1].set_title(f"Target Image-{i+1}")

        ax[i][2].imshow(predicted.detach().squeeze(0).permute(1, 2, 0))
        ax[i][2].set_title(f"Predicted Image-{i+1}")

        for a in ax[i]:
            a.set_xticklabels([])
            a.set_yticklabels([])
    plt.show()

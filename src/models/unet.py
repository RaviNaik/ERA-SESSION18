from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import lightning as L


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, contraction_mode, is_downsample=True):
        super(ContractingBlock, self).__init__()
        self.contraction_mode = contraction_mode
        self.is_downsample = is_downsample

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if self.contraction_mode == "maxpool":
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=2, stride=2
            )

    def forward(self, x):
        x = self.conv_block(x)
        skip = x  # store the output for the skip connection
        if self.is_downsample:
            x = self.downsample(x)
        return x, skip


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_mode):
        super(ExpandingBlock, self).__init__()

        self.expansion_mode = expansion_mode
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if self.expansion_mode == "upsample":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )

    def forward(self, x, skip):
        x = self.upsample(x)
        # concatenate the skip connection
        x = torch.cat((x, skip), dim=1)
        x = self.conv_block(x)

        return x


class UNet(L.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        lr,
        loss,
        contraction_mode,
        expansion_mode,
        epochs,
        maxlr,
        scheduler_steps,
    ):
        super(UNet, self).__init__()

        self.lr = lr
        self.maxlr = maxlr
        self.epochs = epochs
        self.scheduler_steps = scheduler_steps
        self.expansion_mode = expansion_mode
        self.contraction_mode = contraction_mode

        self.loss = loss
        self.contract1 = ContractingBlock(
            in_channels, 64, self.contraction_mode, is_downsample=True
        )
        self.contract2 = ContractingBlock(
            64, 128, self.contraction_mode, is_downsample=True
        )
        self.contract3 = ContractingBlock(
            128, 256, self.contraction_mode, is_downsample=True
        )
        self.contract4 = ContractingBlock(
            256, 512, self.contraction_mode, is_downsample=True
        )
        self.contract5 = ContractingBlock(
            512, 1024, self.contraction_mode, is_downsample=False
        )

        self.expand1 = ExpandingBlock(1024, 512, self.expansion_mode)
        self.expand2 = ExpandingBlock(512, 256, self.expansion_mode)
        self.expand3 = ExpandingBlock(256, 128, self.expansion_mode)
        self.expand4 = ExpandingBlock(128, 64, self.expansion_mode)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 = self.contract4(x)
        x, _ = self.contract5(x)

        # Expanding path
        x = self.expand1(x, skip4)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)
        x = self.final_conv(x)
        return x

    def dice_loss(self, pred, target):
        smooth = 1e-5

        # flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return 1 - dice

    def bce_loss(self, pred, target):
        bce = nn.BCEWithLogitsLoss()(pred, target)
        return bce

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.maxlr,
            steps_per_epoch=self.scheduler_steps,
            epochs=self.epochs,
            pct_start=5 / self.epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        if self.loss == "dice":
            loss = self.dice_loss(out, y)
        else:
            loss = self.bce_loss(out, y)

        self.log(
            name="train_loss",
            value=loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch, batch_idx, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT | None:
        X, y = batch
        out = self(X)
        if self.loss == "dice":
            loss = self.dice_loss(out, y)
        else:
            loss = self.bce_loss(out, y)

        self.log(
            name="val_loss",
            value=loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

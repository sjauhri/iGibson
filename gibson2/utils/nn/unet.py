"""A U-net model with clearer code."""

from .base_unet import BaseUNet
import torch
import torch.nn as nn


class UNet(BaseUNet):
    """Original U-net."""

    def __init__(self, input_channels=1, output_channels=1):
        """Just inherit BaseUnet."""
        super(UNet, self).__init__(input_channels, output_channels)

    def forward(self, img):
        """Forward pass."""
        out0 = self.inc(img)
        out1 = self.down1(out0)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        img = self.up1(out4, out3)
        img = self.up2(img, out2)
        img = self.up3(img, out1)
        img = self.up4(img, out0)
        return (
            self.pos_decoder(img).squeeze(1),
            self.cos_decoder(img).squeeze(1),
            self.sin_decoder(img).squeeze(1),
            self.width_decoder(img).squeeze(1),
            self.graspness_decoder(img).squeeze(1),
            self.bin_classifier(img).squeeze(1)
        )

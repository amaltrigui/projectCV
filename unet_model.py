"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block consisting of the following layers: 
    [3*3 convolution layer --> instance normalization --> LeakyReLU activation--> dropout] * 2
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of the input channels.
            out_chans (int): Number of the output channels.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False), #try with bias 
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),    #try other slopes 
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), #try with bias 
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),   #try other slopes 
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)



class UpConvBlock(nn.Module):
    """
    A Transpose Convolutional Block consisting of : 2x2 deconvolution layer --> instance normalization--> LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of the input channels.
            out_chans (int): Number of the output channels.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), #try other slopes
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

 


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, first_chans, num_blocks, drop_prob):
        """
        Args:
            in_chans (int): Number of the input channels of the whole U-Net model.
            out_chans (int): Number of the output channels of the whole U-Net model.
            first_chans (int): Number of the output channels of the first convolution layer.
            num_blocks (int): Number of the down-sampling blocks = Number of the up-sampling blocks.
            => total number of blocks: 2xnum_blocks + 1 (the bottom block connecting both sides)
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.first_chans = first_chans 
        self.num_blocks = num_blocks
        self.drop_prob = drop_prob

        self.down_layers = nn.ModuleList([ConvBlock(self.in_chans,self.first_chans, drop_prob)])

        ch = self.first_chans 
        for i in range(num_blocks - 1):
            self.down_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2

        self.down_up_conv = ConvBlock(ch, ch * 2, drop_prob)
        ch *= 2

        self.up_layers = nn.ModuleList()
        self.up_deconv_layers = nn.ModuleList()
        for i in range(num_blocks ): 
            self.up_deconv_layers += [UpConvBlock(ch , ch//2)]
            self.up_layers += [ConvBlock(ch, ch//2, drop_prob)]
            ch //= 2

        self.final_layer = nn.Conv2d(ch, self.out_chans, kernel_size=1) 

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)  #try maxpooling 

        output = self.down_up_conv(output)

        # Apply up-sampling layers
        for deconv, conv in zip(self.up_deconv_layers, self.up_layers):
            down_input = stack.pop()
            output = deconv(output)

            diff_H = down_input.size()[2] - output.size()[2]
            diff_W = down_input.size()[3] - output.size()[3]
            output = F.pad(output, [diff_H // 2, diff_H - diff_H // 2, diff_W // 2, diff_W - diff_W // 2])  #check this if bug with padding 
           
            output = torch.cat([output, down_input], dim=1)
            output = conv(output)
        output = self.final_layer(output)

        return output

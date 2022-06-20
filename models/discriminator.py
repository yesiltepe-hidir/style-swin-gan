import torch
import torch.nn as nn
import math

'''
DC-GAN based discriminator
convolutional layers => 1/2 resolution and 2*channels
output signal is of the form: B x 1 x 1 x 1 => discriminate fake/real img
'''

class Discriminator(nn.Module):
    def __init__(self, n_activ_maps=32, n_channels = 3, resolution=256):
        super().__init__()
        
        n_layers = int(math.log2(resolution)-1)
        
        
        net =   [
                    [
                        nn.Conv2d(n_channels, n_activ_maps, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True)
                    ]
                ] +\
                [
                    [
                        nn.Conv2d(n_activ_maps*(2**i), n_activ_maps * (2**(i+1)), 4, 2, 1, bias=False),
                        nn.BatchNorm2d(n_activ_maps * (2**(i+1))),
                        nn.LeakyReLU(0.2, inplace=True),
                    ] for i in range(n_layers-2)
                ] +\
                [
                    [
                        nn.Conv2d(n_activ_maps*(2**(n_layers-2)), 1, 4, 1, 0, bias=False),
                        nn.LeakyReLU(0.2)
                    ]
                ]
        net = sum(net, [])
        self.disc = nn.Sequential(*net)

    def forward(self, input):
        return self.disc(input)
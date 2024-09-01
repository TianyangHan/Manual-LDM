import torch
import torch.nn as nn
from blocks import DownBlocks MidBlock, UpBlock



class VAE(nn.module):
    def __init__(self, in_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.num_down_layers = model_config['num_down_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.down_sample = model_config['down_sample']

        self.attns = model_config['attn_down']

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # upsample reverse downsample
        self.up_sample = list(reversed(self.up_sample)) 

        #encoder
        self.encoder_conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, padding=(1,1))
        self.encoder_layers = nn.ModuleList([])
        self.encoder_mid_layers = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.encoder_layes.append(
                    Down

            )
        



    def encode(self, x):
        mid = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            mid = down(mid)
        for mid in self.encoder_mid:
            mid = self.encoder_mid(mid)

    
    def decode(self, z):



    def forward(self,x):
        z,  = self.encode(x)
        out = self.decode(z)
        return out

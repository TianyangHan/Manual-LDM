import torch
import torch.nn as mm
from models.blocks import get_time_embedding, DownBlock, MidBlock, UpBlockUnet

class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, im_channels, model_config):
        super.__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']

        self.time_projection = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.upsample = list(reversed(self.down_sample))


        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(
                self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, self.dowm_sample[i], self.num_down_layers,
                self.norm_channels, self.num_heads, self.attns[i] 
            ))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(
                self.mid_channels[i], self.num_mid_layers, self.mid_channels[i+1], self.norm_channels, t_emb_dim, self.num_heads,
                
            ))

        self.ups = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.ups.append(
                UpBlockUnet(
                    self.down_channels[i] * 2, self.down_channels[i-1] if i!=0 else self.conv_out_channels,
                    self.t_emb_dim, self.down_sample[i], self.num_up_layers, self.num_down_layers, self.num_heads
                )
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)


    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C x H x W

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.time_projection(t_emb)

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)  # for residual, in advanced append
            out = down(out, t_emb)  
        
        for mid in self.mids:
            out = mid(out, t_emb)

        for up in self.ups:
            down_out = down_outs.pop()  # residual
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
         # out B x C x H x W
        return out
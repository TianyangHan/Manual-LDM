import torch
import torch.nn as nn



class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample,num_layers,
                norm_channels, num_heads, attn, context_dim=None, cross_attn=False):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
                ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels,  out_channels),
                    nn.SiLU(),
                    nn.Conv2d( out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
                ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        if self.t_emb_layers is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.t_emb_dim, out_channels)
                    )
                    for _ in range(num_layers)
                ]
            )
        
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
            )

        if self.cross_attn:
            self.text_emd_layers = nn.ModuleList(
                [
                    nn.Linear(context_dim, out_channels) for _ in range(num_layers)
                ]
            )
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True ) 
                    for _ in range(num_heads)
                ]
            )


    def forward(self,x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)

            if self.t_emb is not None:
                out+=self.t_emb_layers[i](t_emb)[:,:,None,None]

            out = self.resnet_conv_second[i](out)
            out+=self.residual_input_conv[i][resnet_input]

            if self.attn:
                bs, channels, h, w = out.shape
                in_attn = out.reshape(bs, channels, h*w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1,2)
                out_attn = self.attentions[i](in_attn,in_attn,in_attn)
                out_attn = out_attn.transpose(1,2).reshape(bs, channels, h, w)
                out = out+out_attn

            if self.cross_attn:
                assert context is not None, "context should not be None if cross attention"
                bs, channels, h, w = out.shape
                in_attn = out.reshape(bs, channels, h*w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1,2)
                assert context.shape[0] == x.shape(0) and context.shape[-1] == self.context_dim
                context_proj = self.text_emd_layers[i](context)
                out_attn = self.cross_attentions[i](in_attn,context_proj,context_proj)
                out_attn = out_attn.transpose(1,2).reshape(bs, channels, h, w)
                out = out+out_attn

        out = self.down_sample_conv(out)
        return out



class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """

    def __init__(self, in_channels, num_layers, out_channels, norm_channels, t_emb_dim, num_heads, cross_attn=None, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )


        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )


        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])


        self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
            )


        if self.cross_attn:
            self.text_emd_layers = nn.ModuleList(
                [
                    nn.Linear(context_dim, out_channels) for _ in range(num_layers)
                ]
            )
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True ) 
                    for _ in range(num_heads)
                ]
            )


    def forward(self, x, context=None, t_emb=None):
        out = x

        # first resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb is not None:
            out = out + self.text_emb_layers[0](t_emb)[:,:,None, None]
        
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            #attention block
            bs, channels, h, w = out.shape
            in_attn = out.reshape(bs, channels, h*w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1,2)
            out_attn = self.attentions[i](in_attn,in_attn,in_attn)
            out_attn = out_attn.transpose(1,2).reshape(bs, channels, h, w)
            out = out+out_attn

            if self.cross_attn:
                assert context is not None, "context should not be None if cross attention"
                bs, channels, h, w = out.shape
                in_attn = out.reshape(bs, channels, h*w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1,2)
                assert context.shape[0] == x.shape(0) and context.shape[-1] == self.context_dim
                context_proj = self.text_emd_layers[i](context)
                out_attn = self.cross_attentions[i](in_attn,context_proj,context_proj)
                out_attn = out_attn.transpose(1,2).reshape(bs, channels, h, w)
                out = out+out_attn
        
            # resnet block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            if self.t_emb_dim is not None:
                out = out + self.text_emd_layers[i+1](t_emb)[:,:,None,None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        return out
        


class UpBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,num_layers,
                norm_channels, num_heads, attn, context_dim=None, cross_attn=False):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i==0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
                ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels,  out_channels),
                    nn.SiLU(),
                    nn.Conv2d( out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
                ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        if self.t_emb_layers is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.t_emb_dim, out_channels)
                    )
                    for _ in range(num_layers)
                ]
            )
        
        self.up_sample_conv = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=4, stride=2, padding=1) if self.up_sample else nn.Identity()
        

        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)]
        )

        if self.cross_attn:
            self.text_emd_layers = nn.ModuleList(
                [
                    nn.Linear(context_dim, out_channels) for _ in range(num_layers)
                ]
            )
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True ) 
                    for _ in range(num_heads)
                ]
            )


    def forward(self,x, t_emb=None, context=None, out_down=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x,out_down], dim=1)
        out = x
        for i in range(self.num_layers):
            # resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)

            if self.t_emb is not None:
                out+=self.t_emb_layers[i](t_emb)[:,:,None,None]

            out = self.resnet_conv_second[i](out)
            out+=self.residual_input_conv[i][resnet_input]


            bs, channels, h, w = out.shape
            in_attn = out.reshape(bs, channels, h*w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1,2)
            out_attn = self.attentions[i](in_attn,in_attn,in_attn)
            out_attn = out_attn.transpose(1,2).reshape(bs, channels, h, w)
            out = out+out_attn

            if self.cross_attn:
                assert context is not None, "context should not be None if cross attention"
                bs, channels, h, w = out.shape
                in_attn = out.reshape(bs, channels, h*w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1,2)
                assert context.shape[0] == x.shape(0) and context.shape[-1] == self.context_dim
                context_proj = self.text_emd_layers[i](context)
                out_attn = self.cross_attentions[i](in_attn,context_proj,context_proj)
                out_attn = out_attn.transpose(1,2).reshape(bs, channels, h, w)
                out = out+out_attn

        out = self.down_sample_conv(out)
        return out

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

        # Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        ######## Class, Mask and Text Conditioning Config #####
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', None)

        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                validate_text_config(self.condition_config)
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config'][
                    'image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config'][
                    'image_condition_output_channels']


        if self.class_cond:
            # Rather than using a special null class we dont add the
            # class embedding information for unconditional generation
            self.class_emb = nn.Embedding(self.num_classes,
                                          self.t_emb_dim)
        
        if self.image_cond:
            # Map the mask image to a N channel image and
            # concat that with input across channel dimension
            self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
                                          out_channels=self.im_cond_output_ch,
                                          kernel_size=1,
                                          bias=False)
            self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch,
                                            self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.text_cond or self.image_cond or self.class_cond


        self.time_projection = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.upsample = list(reversed(self.down_sample))


        # self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(
                self.down_channels[i], self.down_channels[i+1], self.t_emb_dim, self.dowm_sample[i], self.num_down_layers,
                self.norm_channels, self.num_heads, self.attns[i], cross_attn=self.text_cond, context_dim=self.text_embed_dim
            ))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(
                self.mid_channels[i], self.num_mid_layers, self.mid_channels[i+1], self.norm_channels, t_emb_dim, self.num_heads,
                cross_attn=self.text_cond, context_dim=self.text_embed_dim
            ))

        self.ups = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.ups.append(
                UpBlockUnet(
                    self.down_channels[i] * 2, self.down_channels[i-1] if i!=0 else self.conv_out_channels,
                    self.t_emb_dim, self.down_sample[i], self.num_up_layers, self.num_down_layers, self.num_heads, cross_attn=self.text_cond, 
                    context_dim=self.text_embed_dim
                )
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)


    def forward(self, x, t, cond_input=None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        if self.cond:
            assert cond_input is not None, \
                "Model initialized with conditioning so cond_input cannot be None"
        if self.image_cond:

            
        else:
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
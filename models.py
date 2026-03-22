import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from .metric import MD_distance
import numpy as np
import numbers

EMBEND_DIM = 128 # 64, 96, 128, 256
# NUM_LAYERS = 3
####################################################
def repeat(x):
    if isinstance(x, (tuple, list)):
        return x
    return [x] * 3

def Conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
    )
    return layer

class Mapping(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dim, out_dim, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dim)

        self.conv3x3x3 = nn.Conv3d(1, out_channels=1, kernel_size=3, stride=1,padding=1,bias=False)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x) # 100->128
        x = self.conv3x3x3(x.unsqueeze(1))
        return x

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None

class SSEncoder(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, spectral_kernel_size=(3,1,1), spatial_kernel_size=(1,3,3),
                 spectral_stride=(1,1,1), spatial_stride=(1,1,1),
                 spectral_padding=(1,0,0), spatial_padding=(0,1,1),
                 spatial_dilation=(1,1,1), spectral_dilation=(1,1,1), bias=False): # Add dilation_rate for scale
        super().__init__()
        padding_mode = 'zeros'
        self.spatial_dilation = spatial_dilation
        self.spectral_dilation = spectral_dilation

        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_1, spatial_kernel_size, spatial_stride,
                      padding=(0, spatial_padding[1]*spatial_dilation[1], spatial_padding[2]*spatial_dilation[2]),
                      dilation=spatial_dilation, bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels_1, out_channels_2, spatial_kernel_size, (1,1,1),
                      padding=(0, spatial_padding[1]*spatial_dilation[1], spatial_padding[2]*spatial_dilation[2]),
                      dilation=spatial_dilation, bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels_2, out_channels_1, spatial_kernel_size, (1,1,1),
                      padding=(0, spatial_padding[1]*spatial_dilation[1], spatial_padding[2]*spatial_dilation[2]),
                      dilation=spatial_dilation, bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )

        self.spectral_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_1, spectral_kernel_size, spectral_stride,
                      padding=(spectral_padding[0]*spectral_dilation[0], 0, 0),
                      dilation=spectral_dilation, bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels_1, out_channels_2, spectral_kernel_size, (1,1,1),
                      padding=(spectral_padding[0]*spectral_dilation[0], 0, 0),
                      dilation=spectral_dilation, bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels_2, out_channels_1, spectral_kernel_size, (1,1,1),
                      padding=(spectral_padding[0]*spectral_dilation[0], 0, 0),
                      dilation=spectral_dilation, bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )

        # self.fuse_conv0 = nn.Sequential( # spatial_conv spectral_conv 1->4->8->8->1
        #     nn.Conv3d(out_channels_2 * 2, out_channels_1, kernel_size=1, stride=1, padding=0, bias=bias),
        #     nn.BatchNorm3d(out_channels_1),
        #     nn.Conv3d(out_channels_1, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        # )
        self.fuse_conv1 = nn.Sequential( # spatial_conv spectral_conv 1->4->8->4->1
            # nn.Conv3d(out_channels_1 * 2, out_channels_1, kernel_size=1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm3d(out_channels_1),
            nn.Conv3d(out_channels_1*2, in_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        # (B, 1, Spectral_Dim, H, W)
        x0 = x
        spatial_y = self.spatial_conv(x)
        spectral_y = self.spectral_conv(x)

        fused_features = self.fuse_conv1(torch.cat([spatial_y, spectral_y], dim=1))
    
        out = x0 + fused_features
        return out

class RouterEncoder(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.scale_configs = [
            ((1, 1, 1), (1, 1, 1)),  
            ((1, 2, 2), (1, 1, 1)),  
            ((1, 1, 1), (2, 1, 1)),  
            ((1, 2, 2), (2, 1, 1)),  
            ((1, 1, 1), (3, 1, 1)),  
            ((1, 2, 2), (3, 1, 1)), 
         ]

        self.block_count = len(self.scale_configs)
        
        self.blocks = nn.ModuleList([
            SSEncoder(in_channels, out_channels_1, out_channels_2, 
                     spatial_dilation=spatial_dilation,  
                     spectral_dilation=spectral_dilation,  
                     bias=bias)
            for spatial_dilation, spectral_dilation in self.scale_configs
        ])

        self.routing = nn.Sequential(
            nn.Conv2d(EMBEND_DIM, 32, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)), 
        )

        self.l1 = nn.Linear(in_features=32, out_features=self.block_count) 

    def forward(self, x):
        # x input: (B, 1, Spectral_Dim, H, W) after mapping and reshaping
        x_origin = x
        batch_size = x.shape[0]
        x_for_routing_2d = x.squeeze(1) # (B, EMBEND_DIM, H, W)
        routing_vector = self.routing(x_for_routing_2d).reshape(batch_size, -1)
        routing_vector = torch.sigmoid(self.l1(routing_vector))
        routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * (self.block_count / 2)
        routing_vector = torch.clamp(routing_vector, 0, 1)
        ref = RoundSTE.apply(routing_vector) # (B, block_count) with 0s and 1s

        current_feature = x
        for i in range(self.block_count):
            gate = ref[:, i].view(batch_size, 1, 1, 1, 1)

            processed_feature = self.blocks[i](current_feature) # Output is (B, 1, EMBEND_DIM, H, W) residual
            # gate operation
            current_feature = gate * processed_feature + (1 - gate) * current_feature

        # final_feature = current_feature + x_origin # resdial
        final_feature = current_feature
        
        return final_feature

class feature_encode(nn.Module):
    def __init__(self, src_dim, tar_dim, in_dim = EMBEND_DIM):
        super(feature_encode, self).__init__()
        self.in_dim = in_dim
        self.source_mapping = Mapping(src_dim, self.in_dim)
        self.target_mapping = Mapping(tar_dim, self.in_dim)
        self.routernet = RouterEncoder(1,4,8)

        self.flatten_out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        # embedding_feature = flatten.view(flatten.shape[0],-1)

    def forward(self, x, domain = 'source'): 
        """
        support: (b,c,h,w)
        """
        if domain == 'source':
            x = self.source_mapping(x)
        elif domain == 'target':
            x = self.target_mapping(x)
        out = self.routernet(x)
        out = self.flatten_out(out.squeeze(1))
        return out

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     support_f = torch.rand(45, 128, 9, 9)
#     network = feature_encode(128,128)
#     print(get_parameter_number(network))
#     out = network(support_f)
#     print(out.shape)

#from thop import profile
#if __name__ == '__main__':
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    supports = torch.rand(1, 100, 9, 9).to(device)
#    support_labels = torch.randint(0, 9, (1,)).to(device)
#
#    feature_encoder = feature_encode(100, 128).to(device)
#    print(get_parameter_number(feature_encoder))
#   flops, params = profile(feature_encoder, inputs=(supports, "source"))
#    print(f"FLOPs: {flops / 1e6} MFLOPs")
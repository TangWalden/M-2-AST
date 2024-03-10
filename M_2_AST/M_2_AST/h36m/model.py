import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# Cyber Fc
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
from torch import Tensor
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

# Dynamic Block
from dynamic_mixer import DynaMixerBlock
from dynamic_mixer import DynaMixerOperation
from einops import rearrange

import random

# seed
seed = 300
random.seed(seed)  # random
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # torch+CPU
torch.cuda.manual_seed(seed)  # torch+GPU

class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


    
    
def mish(x):
    return (x*torch.tanh(F.softplus(x)))


    
    
class MlpBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, mlp_input_dim, mlp_bn_dim, activation='gelu', regularization=0, initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        #self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'mish':
            self.act1 = mish #nn.Mish()
        else:
            raise ValueError('Unknown activation function type: %s'%activation)
            
            
        

    def forward(self, x):
        o = x
        x = self.fc1(x)  
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.fc2(x)  
        if self.reg2 is not None:
            x = self.reg2(x)

        return x + o


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.imagesize = (self.seq_len, self.seq_len)
        self.channels_mlp_dim_2 = hidden_dim
        self.d_token = 1
        self.d_channel = 1

        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim, activation=activation, regularization=regularization, initialization=initialization)
        self.mlp_block_token_mixing_DynaMixerOperation = DynaMixerOperation(self.seq_len, self.hidden_dim, self.d_token)
        self.mlp_block_channel_mixing_DynaMixerOperation = DynaMixerOperation(22, self.seq_len, self.d_channel)
        
        self.mlp_block_channel_mixing_base_MLPmixer = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len, activation=activation, regularization=regularization, initialization=initialization)
        self.mlp_block_channel_mixing = CycleMLP(self.channels_mlp_dim_2, qkv_bias=False)
       
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN = nn.LayerNorm(self.hidden_dim)


        self.fc_saptial_in = nn.Linear(1, self.hidden_dim)
        self.fc_saptial_out = nn.Linear(self.hidden_dim, 1)
        self.fc_in = nn.Linear(self.hidden_dim, 22)
        self.fc_out = nn.Linear(22, self.hidden_dim)

        self.fc_ = nn.Linear(10, 10)

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels
        batch_size, _, _ = x.shape

        # print(x.shape,'1')   # [50, 10, 66]   
        y = self.LN(x)    
        y = y.transpose(1, 2)  # y = [50, 50, 10]
        z = y
        z = self.mlp_block_token_mixing(z)  # spatial mixer
        y = self.mlp_block_token_mixing_DynaMixerOperation(y)
        y = y.transpose(1, 2)
        z = z.transpose(1, 2)
        
        if self.use_se:
            y = self.se(y)
            z = self.se(z)

        x = x + y + z

        # print(x.shape,'2') # [50, 10, 66]
                
        y = self.LN(x) 
        z = y
        y = y.unsqueeze(3)
        print(y.shape)
        y = self.fc_saptial_in(y)
        y = self.mlp_block_channel_mixing(y)
        y = self.fc_saptial_out(y)
        y = y.squeeze(3)
        y = y.reshape(batch_size, self.seq_len, self.hidden_dim)
        z = self.mlp_block_channel_mixing_base_MLPmixer(z)

        if self.use_se:
            y = self.se(y)
            z = self.se(z)
        # x = x + z
        x = x + y + z
        
        # # print(x.shape,'3') # [50, 10, 66]
        y = self.LN(x)
        y = self.fc_in(y)
        y = self.mlp_block_channel_mixing_DynaMixerOperation(y)
        y = self.fc_out(y) 

        if self.use_se:
            y = self.se(y)
         

        x = x + y
        # print(x.shape,'4') # [50, 10, 66]


        #         # print(x.shape,'1')   # [50, 10, 66]   
        # y = self.LN(x)    
        # y = y.transpose(1, 2)  # y = [50, 50, 10]
        # z = y
        # z = self.fc_(z)  # spatial mixer
        # # y = self.mlp_block_token_mixing_DynaMixerOperation(y)
        # # y = y.transpose(1, 2)
        # z = z.transpose(1, 2)
        
        # if self.use_se:
        #     # y = self.se(y)
        #     z = self.se(z)

        # x = x + z
        
        return x


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim, seq_len, pred_len, activation='gelu', mlp_block_type='normal',
                 regularization=0, input_size=51, initialization='none', r_se=4, use_max_pooling=False, use_se=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.input_size = input_size    
        self.activation = activation
        self.Mixer_Block = nn.ModuleList(MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, self.seq_len, self.hidden_dim, activation=self.activation, 
                                            regularization=regularization, initialization=initialization, r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se) for _ in range(num_blocks))
        self.LN = nn.LayerNorm(self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.num_classes)
        self.pred_len = pred_len
        self.conv_out = nn.Conv1d(self.seq_len, self.pred_len, 1, stride=1)

        self.LN_in = nn.Linear(self.num_classes, self.hidden_dim)

    def get_dct_matrix(self, x): 
        dct_m = np.eye(x)   
        for k in np.arange(x):
            for i in np.arange(x):
                w = np.sqrt(2 / x)  
                if k == 0:
                    w = np.sqrt(1 / x)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / x) #(25, 25)
        idct_m = np.linalg.inv(dct_m) 
        return dct_m, idct_m                

    def forward(self, x): 
        dct_m,idct_m = self.get_dct_matrix(self.seq_len)
        dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
        idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)       
        x = torch.matmul(dct_m[:, :, :self.seq_len], x)

        y = self.LN_in(x) #
        for mb in self.Mixer_Block:
            y = mb(y)
        y = self.LN(y)

        y = torch.matmul(idct_m[:, :self.seq_len, :], y)       
        out = self.fc_out(self.conv_out(y))

        return out
    
# CycleFC

class CycleFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels  # 22
        self.out_channels = out_channels  # 22
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups  # 1

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset



    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()  # [50,3,10,22]
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation, mask=None)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)



class CycleMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.LN = nn.LayerNorm(dim)


        self.sfc_h = CycleFC(dim, dim, (1, 10), 1, 0) # dim 22
        self.sfc_w = CycleFC(dim, dim, (10, 1), 1, 0)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        y = self.LN(x)
        B, H, W, C = y.shape # 50 10 22 3 torch.Size([50, 10, 22, 3])
        h = self.sfc_h(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(h.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        c = self.mlp_c(w)

        # c = self.LN(c)

        x = self.proj(c)
        x = self.proj_drop(x)

        return x + y


import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

class DynaMixerOperation(nn.Module):

    def __init__(self, N, D, d=18):  # d = 2
        super().__init__()

        self.N = N
        self.D = D
        self.d = d

        self.fc_ND = nn.Linear(D, d)
        self.fc_Nd = nn.Linear(N*d, N*N)
        self.fc_out = nn.Linear(D, D)
        self.softmax = nn.Softmax(dim=2)
        self.LN = nn.LayerNorm(N)

    def forward(self, input):

        B, D, N = input.shape

        # Dynamic mixing matrix generation
        input = rearrange(input, 'b d n -> b n d')

        p = self.fc_ND(input)
        p = p.reshape(-1, 1, N*self.d)
        p = self.fc_Nd(p)
        p = p.reshape(-1, N, N)
        p = self.softmax(p)  # p [50 10 10]
        p = self.LN(p)
        p_save = p.cpu().detach().numpy()
        p_save = np.mean(p_save, axis=0)
        # np.save('/content/drive/MyDrive/MotionMixer-main/MotionMixer-main/h36m/weight',p_save)
        out = torch.matmul(p, input)  # input[50, 10, 66]
        out = self.fc_out(out)
        out = rearrange(out, 'b n d -> b d n')
        return out

class DynaMixerBlock(nn.Module):

    def __init__(self, channels=66, imagesize=(10, 10)):
        super().__init__()

        h_size = imagesize[0]  # 10
        w_size = imagesize[1]  # 10
        self.dynamixer_op_h = DynaMixerOperation(w_size, channels)
        self.dynamixer_op_w = DynaMixerOperation(h_size, channels)
        self.proj_c = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_o = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, input_1):

        b, c, h, w = input_1.shape

        # row mixing
        Y_h = input_1
        for i in range(h):
            # print(input[:, :, i, :].shape,'input[:, :, i, :]')
            Y_h[:, :, i, :] = self.dynamixer_op_h(input_1[:, :, i, :])     # (b c w)

        # column mixing
        Y_w = input_1
        for i in range(w):
            Y_w[:, :, :, i] = self.dynamixer_op_w(input_1[:, :, :, i])     # (b c h)

        # channel mixing
        Y_c = self.proj_c(input_1)
        Y_out = Y_h + Y_w + Y_c

        return self.proj_o(Y_out)


if __name__ == '__main__':
    input = torch.rand([50, 66, 10, 10])
    a = input[:, :, :, ]
    SEED = 42
    torch.manual_seed(SEED)

    model = DynaMixerBlock()
    input = torch.rand([50, 66, 10, 10])  # (b c h w)

    out = model(input)
    print(out.shape)

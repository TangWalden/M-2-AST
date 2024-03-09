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
        self.d_token = 22
        self.d_channel = 10

        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim, activation=activation, regularization=regularization, initialization=initialization)
        self.mlp_block_token_mixing_DynaMixerOperation = DynaMixerOperation(self.seq_len, self.hidden_dim)
        self.mlp_block_channel_mixing_DynaMixerOperation = DynaMixerOperation(self.hidden_dim, self.seq_len)
        self.mlp_block_channel_mixing_base_MLPmixer = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len, activation=activation, regularization=regularization, initialization=initialization)
        self.mlp_block_channel_mixing = CycleMLP(self.channels_mlp_dim_2, qkv_bias=False)
        self.dynamic_channel = spatialFC(self.seq_len, self.channels_mlp_dim_2)
        
        self.use_se = use_se
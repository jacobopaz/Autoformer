class Config():

    def __init__(
        self,
        embed_dim: int = 20,
        att_heads: int = 5,
        att_dropout: float = 0.1,
        autocorr_factor: int = 3,
        use_bias: bool = True,
        activation: str = "relu",
        activ_dropout: float = 0.1,
        ffn_dim: int = 64,
        dropout: float = 0.1,
        n_enc_layers: int = 5,
        kernel_size: int = 25,
        max_wavelength: int = 10000,
    ):
        
        self.embed_dim = embed_dim
        self.att_heads = att_heads
        self.att_dropout = att_dropout
        self.autocorrelation_factor = autocorr_factor
        self.use_bias = use_bias
        self.activation = activation
        self.activation_dropout = activ_dropout
        
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.n_enc_layers = n_enc_layers

        self.kernel_size = kernel_size

        self.max_wavelength = max_wavelength
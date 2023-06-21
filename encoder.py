import tensorflow as tf
from keras.models import Sequential
from keras.layers import Layer, Dropout, EinsumDense, LayerNormalization, Dense
from keras.layers import activation
from decomposition import DecompositionLayer
from positional_encoding import SinePositionEncoding
from configuration.configuration import Config


class Dense2Embed(Layer):
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        super().__init__()
    
    def build(self, input_shape):
        self.embed_layer = Dense(self.embed_dim)
        super().build(input_shape)

    def call(self, x):
        return self.embed_layer(x)
    


class SumPositionEncoding(Layer):
    def __init__(self, max_wavelength: int):
        self.max_wavelength = max_wavelength
        super().__init__()
        
    def call(self, x):
        positional_encoded = SinePositionEncoding(self.max_wavelength)(x)
        return x + positional_encoded


class AutoFormerLayerNormalization(Layer):

    def __init__(self):
        self.ln = LayerNormalization()
        super().__init__()

    def call(self, x):
        x_ = self.ln(x)
        bias = tf.repeat(tf.math.reduce_mean(x, axis=1, keepdims=True), repeats=[x.shape[1]], axis=1)
        return x_ - bias 
 

class AutoCorrelation(Layer):

    """
    Custom layer for calculating autocorrelation-based attention.

    Args:
        embed_dim (int): The embedding dimension.
        n_heads (int): The number of attention heads.
        dropout (float, optional): Dropout rate (default: 0.0).
        autocorr_factor (int, optional): Autocorrelation factor (default: 3).
        use_bias (bool, optional): Whether to use bias in dense layers (default: True).
    """

    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        dropout: float = 0.0, 
        autocorr_factor: int = 3, 
        use_bias: bool = True, 
    ): 
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dropout_layer = Dropout(dropout)
        self.autocorr_factor = autocorr_factor
        self.use_bias = use_bias
        super().__init__()

        assert self.n_heads > 0, "Number of attention heads must be greater than 0."
        
    
    def build(self, input_shape):
        """
        Builds the internal layers of the AutoCorrelation layer.

        Args:
            input_shape: The shape of the input tensor.

        Raises:
            AssertionError: If the input shape does not have 3 dimensions.
        """
        assert len(input_shape) == 3, "Input shape must have 3 dimensions: (batch_size, sequence_length, input_dim)"

        bias_axes = 'uh' if self.use_bias else None
        self.wk = EinsumDense('btj,juh->btuh', output_shape=(None, self.embed_dim, self.n_heads), bias_axes=bias_axes)
        self.wq = EinsumDense('btj,juh->btuh', output_shape=(None, self.embed_dim, self.n_heads), bias_axes=bias_axes)
        self.wv = EinsumDense('btj,juh->btuh', output_shape=(None, self.embed_dim, self.n_heads), bias_axes=bias_axes)
        
        self._softmax = activation.Softmax(axis=-2)

        bias_axes = 'f' if self.use_bias else None
        self.linear = EinsumDense('btjh,jhf->btf', output_shape=[None, self.embed_dim], bias_axes=bias_axes)

    
    def call(self, hidden_states, training=None):
        """
        Performs the forward pass of the AutoCorrelation layer.

        Args:
            hidden_states: The input tensor --> Input shape: BatchSize x seq_len x Channel (channel and embedding dim is equivalent)
            training (bool, optional): Whether the model is in training mode (default: True).

        Returns:
            att_out: The output tensor.

        Raises:
            ValueError: If the shape of the attention weights is incorrect.
        """

        bsize, seq_len, _ = hidden_states.shape

        keys = self.wk(hidden_states)
        queries = self.wq(hidden_states)
        values = self.wv(hidden_states)


        queries_time_length = queries.shape[1]
        values_time_length = values.shape[1]
        if queries_time_length > values_time_length:
            queries = queries[:, : (queries_time_length - values_time_length), :, :]
            zeros = tf.zeros_like(queries).float()
            values = tf.concat([values, zeros], axis=1)
            keys = tf.concat([keys, zeros], axis=1)
        else:
            values = values[:, :queries_time_length, :]
            keys = keys[:, :queries_time_length, :]
            

        # tf.signal.rfft applies over the most inner dimension, so we must transpose to get the time dimension in that position (axis = -1).
        queries_fft = tf.signal.rfft(tf.transpose(queries, [0, 2, 3, 1]), tf.constant([seq_len], dtype=tf.int32))
        keys_fft = tf.signal.rfft(tf.transpose(keys, [0, 2, 3, 1]), tf.constant([seq_len], dtype=tf.int32))

        channel = keys.shape[2]

        att_weights = queries_fft * tf.math.conj(keys_fft)
        att_weights = tf.signal.irfft(att_weights, tf.constant([seq_len], dtype=tf.int32))
        att_weights = tf.transpose(att_weights, [0, 3, 1, 2])
        
        if att_weights.shape != (bsize, seq_len, channel, self.n_heads):
            raise ValueError(
                f"Attention weights should be of size {(bsize, seq_len, channel, self.n_heads)}, but is"
                f" {att_weights.shape}"
            )
        

        time_len = values.shape[1]
        autocorrelations = att_weights

        # top k autocorrelations delays
        top_k = int(self.autocorr_factor * tf.math.log(tf.constant([time_len], dtype=tf.float32)))
        autocorrels_mean_on_head = tf.math.reduce_mean(autocorrelations, axis=(-1))  # batch size x seq_len x channel
        if training:
            autocorrels_mean_on_bsize = tf.math.reduce_mean(autocorrels_mean_on_head, axis=0)
            _, top_k_delays_index = tf.math.top_k(tf.transpose(autocorrels_mean_on_bsize, [1, 0]), top_k)
            top_k_delays_index = tf.transpose(top_k_delays_index, [1, 0])
            top_k_autocorrels = tf.stack([
                                    autocorrels_mean_on_head[:, top_k_delays_index[i, j], j] 
                                    for i in range(top_k) for j in range(channel)
                                ], axis=1)
            top_k_autocorrels = tf.reshape(top_k_autocorrels, [bsize, top_k, channel]) 
        else:
            top_k_autocorrels, top_k_delays_index = tf.math.top_k(
                tf.transpose(autocorrels_mean_on_head, [0, 2, 1]), top_k
            )
            top_k_delays_index = tf.transpose(top_k_delays_index, [0, 2, 1])
            top_k_autocorrels = tf.transpose(top_k_autocorrels, [0, 2, 1])
    
        top_k_autocorrels = self._softmax(top_k_autocorrels)  # batch size x top_k x channel

        if not training:
            # used for compute values_roll_delay in inference
            tmp_values = tf.repeat(values, [2], axis=1)
            init_idx = tf.repeat(
                    tf.repeat(
                            tf.repeat(
                                tf.reshape(
                                    tf.range(time_len), [1, -1, 1, 1]
                                ), [bsize], axis=0
                            ), [channel], axis=2
                        ), [self.n_heads], axis=-1
                ) 
          
        delays_agg = tf.zeros_like(values, dtype=tf.float32)  # batch size x time_length x channel x n_heads 
        
        for i in range(top_k):    
            if not training:
                values_roll_delay = []
                for j in range(channel):
                    tmp_delay = init_idx + tf.repeat(tf.reshape(top_k_delays_index[:, i, :], [bsize, 1, channel, 1]), [self.n_heads], axis=-1)       
                    values_roll_delay_ch = tf.gather(tmp_values[:, :, j, :], axis=1, indices=tmp_delay[:, :, j, 0], batch_dims=1)
                    values_roll_delay.append(values_roll_delay_ch)

                values_roll_delay = tf.stack(values_roll_delay, axis=2)

            else:   
                values_roll_delay = []
                for j in range(channel): 
                    values_roll_delay.append(tf.roll(values[:, :, j, :], shift=-int(top_k_delays_index[i, j]), axis=1))
                
                values_roll_delay = tf.stack(values_roll_delay, axis=2)

            # aggregation
            top_k_autocorrels_at_delay = (
                tf.repeat(
                    tf.repeat(
                        tf.reshape(top_k_autocorrels[:, i, :], [bsize, 1, channel, 1]), [seq_len], axis=1
                    ), [self.n_heads], axis=-1
                )
            )

        delays_agg += values_roll_delay * top_k_autocorrels_at_delay
        
        assert (
            delays_agg.shape == (bsize, seq_len, self.embed_dim, self.n_heads)
        ), f"`delays_agg` should be of size {(bsize, seq_len, self.embed_dim, self.n_heads)}, but is"
        f" {delays_agg.shape}"

        att_out = self.linear(delays_agg)
        return att_out
        


class AutocorrEncoderLayer(Layer):
    """
    AutocorrEncoderLayer is a layer used in the Autocorrelation Transformer encoder.
    It consists of frequency domain self-attention (autocorrelation), feedforward neural network, and residual connections.

    Args:
        config (Config): Configuration object specifying various hyperparameters.

    Attributes:
        embed_dim (int): Dimensionality of the input embeddings.
        dropout (float): Dropout rate to apply to the intermediate outputs.
        activation (str or Callable): Activation function to use in the feedforward neural network.
        activation_dropout (float): Dropout rate to apply to the activation outputs.
        config (Config): Configuration object specifying various hyperparameters.
        self_att (AutoCorrelation): Autocorrelation self-attention layer.
        ln (LayerNormalization): Layer normalization layer.
        dropout_layer1 (Dropout): Dropout layer for the first dropout.
        dropout_layer2 (Dropout): Dropout layer for the second dropout.
        feedforward (Sequential): Feedforward neural network layer ( 2 Denses and Dropout).
        final_ln (AutoFormerLayerNormalization): Layer normalization layer after the feedforward layer.
        decomp_layer1 (DecompositionLayer): Decomposition layer 1.
        decomp_layer2 (DecompositionLayer): Decomposition layer 2.

    Methods:
        build(input_shape): Builds the layer by initializing the sub-layers and setting their shapes.
        call(hidden_states): Performs the forward pass of the layer.

    """

    def __init__(self, config: Config):
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout

        self.activation = config.activation
        self.activation_dropout = config.activation_dropout

        self.config = config
        super().__init__()


    def build(self, input_shape):
        """
        Builds the layer by initializing the sub-layers and setting their shapes.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.self_att = AutoCorrelation(
            embed_dim=self.embed_dim,
            n_heads=self.config.att_heads,
            dropout=self.config.att_dropout,
            autocorr_factor=self.config.autocorrelation_factor,
            use_bias=self.config.use_bias
        )

        self.ln = LayerNormalization()
        self.dropout_layer1 = Dropout(self.dropout)
        self.dropout_layer2 = Dropout(self.dropout)

        self.feedforward = Sequential([
                    Dense(self.config.ffn_dim, activation=self.activation), 
                    Dropout(self.dropout),
                    Dense(self.embed_dim, activation=self.activation)
                ])
        
        self.final_ln = AutoFormerLayerNormalization()
        self.decomp_layer1 = DecompositionLayer(self.config)
        self.decomp_layer2 = DecompositionLayer(self.config)
        
        super().build(input_shape)

    
    def call(self, hidden_states):
        """
        Performs the forward pass of the layer.

        Args:
            hidden_states: Input hidden states.

        Returns:
            hidden_states: Output hidden states after the forward pass.
        """
        res = hidden_states
        hidden_states = self.self_att(hidden_states=hidden_states)

        hidden_states = self.dropout_layer1(hidden_states)
        hidden_states = res + hidden_states

        hidden_states = self.ln(hidden_states)
        hidden_states, _ = self.decomp_layer1(hidden_states)

        res = hidden_states
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout_layer2(hidden_states)
        hidden_states = hidden_states + res

        hidden_states, _ = self.decomp_layer2(hidden_states)
        hidden_states = self.final_ln(hidden_states)
        return hidden_states



class AutoFormerEncoder(Layer):
    """
    AutoFormerEncoder is an encoder module used in the AutoFormer architecture.
    It consists of an embedding layer, sum position encoding (sine positional encoding), layer normalization, dropout layer,
    and a stack of AutocorrEncoderLayer layers.

    Args:
        config (Config): Configuration object specifying various hyperparameters.

    Attributes:
        dropout (float): Dropout rate to apply to the intermediate outputs.
        config (Config): Configuration object specifying various hyperparameters.
        embedding (CustomEmbedding): Custom embedding layer.
        sum_pos_encoding (SumPositionEncoding): Sum position encoding layer.
        ln (LayerNormalization): Layer normalization layer.
        dropout_layer (Dropout): Dropout layer.
        enc_layers (Sequential): Sequential stack of AutocorrEncoderLayer layers.

    Methods:
        build(input_shape): Builds the layer by initializing the sub-layers and setting their shapes.
        call(hidden_states): Performs the forward pass of the layer.
    """

    def __init__(self, config: Config): 
        self.dropout = config.dropout 
        self.config = config
        super().__init__()  

    def build(self, input_shape):
        """
        Builds the layer by initializing the sub-layers and setting their shapes.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.embedding = Dense2Embed(self.config.embed_dim)
        self.sum_pos_encoding = SumPositionEncoding(self.config.max_wavelength)
        self.ln = LayerNormalization()
        self.dropout_layer = Dropout(self.dropout)
        self.enc_layers = Sequential([AutocorrEncoderLayer(self.config) for layer in range (self.config.n_enc_layers)])
        super().build(input_shape)

    def call(self, hidden_states):
        """
        Performs the forward pass of the layer.

        Args:
            hidden_states: Input hidden states.

        Returns:
            hidden_states: Output hidden states after the forward pass.
        """
        hidden_states = self.embedding(hidden_states)
        hidden_states = self.sum_pos_encoding(hidden_states)
        hidden_states = self.ln(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = self.enc_layers(hidden_states)
        return hidden_states
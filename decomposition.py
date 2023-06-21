import tensorflow as tf
from keras.layers import Layer, AveragePooling1D
from configuration.configuration import Config


class DecompositionLayer(Layer):
    """
    Custom layer for signal decomposition into trend and seasonal components.
    
    This layer takes an input signal and decomposes it into two components:
    - Trend component: Represents the underlying long-term trend in the signal.
    - Seasonal component: Represents the seasonal fluctuations in the signal.
    
    The decomposition is performed using a sliding average filter. The input signal is padded
    on both ends to ensure that the filter can be applied to the entire signal.
    
    Args:
        kernel_size (int): Size of the sliding average filter.
            Must be an odd number.
    
    Raises:
        AssertionError: If the kernel_size is larger than the input shape or not an odd number.
    
    Attributes:
        kernel_size (int): Size of the sliding average filter.
        avg (AveragePooling1D): Average pooling layer used for trend component calculation.
    
    Example:
        # Create a DecompositionLayer with kernel size of 5
        decomposition_layer = DecompositionLayer(kernel_size=5)
        # Apply the layer to an input signal
        trend, seasonal = decomposition_layer(input_signal)
    """

    def __init__(self, config: Config):
        self.kernel_size = config.kernel_size
        super().__init__()

    def build(self, input_shape):
        """
        Builds the decomposition layer.
        
        This method creates the average pooling layer used for trend component calculation.
        It also performs assertions to ensure that the kernel_size is valid for the input shape.
        
        Args:
            input_shape (tuple): Shape of the input tensor.
        
        Raises:
            AssertionError: If the kernel_size is larger than the input shape or not an odd number.
        """
        self.avg = AveragePooling1D(self.kernel_size, strides=1, padding='valid', data_format='channels_last')
        assert self.kernel_size <= input_shape[1], "Kernel size must be equal or smaller than input_shape."
        assert self.kernel_size % 2 != 0, "Kernel size must be an odd number."
        super().build(input_shape)

    def call(self, x):
        """
        Applies the decomposition layer to the input signal.
        
        This method performs the signal decomposition by applying the sliding average filter.
        It pads the input signal on both ends to ensure that the filter can be applied to the entire signal.
        
        Args:
            x (tf.Tensor): Input signal tensor.
        
        Returns:
            tuple: A tuple containing the trend and seasonal components.
                - trend (tf.Tensor): Trend component of the input signal.
                - seasonal (tf.Tensor): Seasonal component of the input signal.
        """
        n_pads = (self.kernel_size - 1) // 2
        front_pad = tf.repeat(x[:, 0:1, :], [n_pads], axis=1)
        end_pad = tf.repeat(x[:, -1:, :], [n_pads], axis=1)
        x_padded = tf.concat([front_pad, x, end_pad], axis=1)

        x_trend = self.avg(x_padded)
        x_seasonal = x - x_trend

        return x_seasonal, x_trend

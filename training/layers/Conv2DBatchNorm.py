import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization, Conv2D, LeakyReLU

class Conv2DBatchNorm(Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, batch_norm= True):
        super(Conv2DBatchNorm, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        
    def build(self, input_shape):
        self.conv = Conv2D(filters= self.filters, kernel_size= self.kernel_size, strides= self.strides, padding= self.padding, use_bias= self.use_bias)
        if self.batch_norm:
            self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU(alpha= 0.2)
        
        super(Conv2DBatchNorm, self).build(input_shape)
    
    def call(self, input_tensor):
        x = input_tensor
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.leaky_relu(x)
        
        return x
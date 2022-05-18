import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

from layers import Conv2DBatchNorm

class Discriminator(Model):
    def __init__(self, name, **kwargs):
        super(Discriminator, self).__init__(name, **kwargs)
    
    def build(self, input_shape):
        self.conv1 = Conv2DBatchNorm(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)
        
        self.conv_bn_1 = Conv2DBatchNorm(filters=64*2, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)
        self.conv_bn_2 = Conv2DBatchNorm(filters=64*4, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)        
        self.conv_bn_3 = Conv2DBatchNorm(filters=64*8, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)        
        self.conv_bn_4 = Conv2DBatchNorm(filters=64*8, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)        
        self.conv_bn_5 = Conv2DBatchNorm(filters=64*4, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)        
        self.conv_bn_6 = Conv2DBatchNorm(filters=64*2, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)
        self.conv_bn_7 = Conv2DBatchNorm(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False, batch_norm= False)  
        
        self.conv2 = Conv2D(1, (3, 3), strides=(4, 4), padding="same", use_bias= False, activation= 'sigmoid')
        super(Discriminator, self).build(input_shape)
    
    def call(self, input_tensor):
        x = input_tensor
        x = self.conv1(x)
        
        x = self.conv_bn_1(x)
        x = self.conv_bn_2(x)
        x = self.conv_bn_3(x)
        x = self.conv_bn_4(x)
        x = self.conv_bn_5(x)        
        x = self.conv_bn_6(x)
        x = self.conv_bn_7(x)
        
        x = self.conv2(x)
        return x
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, Dropout, 
                                     Flatten, Embedding, multiply, Input)

class CDiscriminator(Model):
    def __init__(self, image_size, num_classes):
        super(CDiscriminator, self).__init__()
        self.img_shape, self.num_classes = image_size, num_classes
        
    def build(self, input_shape):
        self.seq_model = Sequential(
            [
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dropout(0.4),
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dropout(0.4),
                Dense(1, activation='sigmoid')
            ]
        )
        super(CDiscriminator, self).build(input_shape)
        
    def call(self, input_tensor):
        img, label = input_tensor
        assert img.shape[1:] == self.img_shape, f"Received input shape: {img.shape[1:]} does not match image shape: {self.img_shape} in Discriminator Call"
        
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])
        
        output = self.seq_model(model_input)
        return output
    
def build_discriminator(img_shape, num_classes):

    model = Sequential()

    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)
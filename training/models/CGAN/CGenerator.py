import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, Conv2DTranspose, 
                                     Conv2D, Reshape, Flatten, Embedding, 
                                     multiply, Input, InputLayer)

class CGenerator(Model):
    def __init__(self, latent_dim, num_classes):
        super(CGenerator, self).__init__()
        self.latent_dim, self.num_classes = latent_dim, num_classes
        
    def build(self, input_shape):
        self.seq_model = Sequential(
            [
                Dense(16 * 16 * (self.latent_dim + self.num_classes)),
                LeakyReLU(alpha=0.2),
                Reshape((16, 16, (self.latent_dim + self.num_classes))),
                Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2D(3, (7, 7), padding="same", activation="tanh"),    
            ]
        )
        super(CGenerator, self).build(input_shape)
        
    def call(self, input_tensor):
        noise, label = input_tensor
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        
        output = self.seq_model(model_input)
        return output
    
def build_generator(latent_dim, num_classes):

    model = Sequential(
    [
        InputLayer(latent_dim,),
        Dense(16 * 16 * (latent_dim + num_classes)),
        LeakyReLU(alpha=0.2),
        Reshape((16, 16, (latent_dim + num_classes))),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(3, (7, 7), padding="same", activation="tanh"),
    ],
    name="generator",
    )

    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)
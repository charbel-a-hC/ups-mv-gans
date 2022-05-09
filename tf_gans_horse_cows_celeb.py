import tensorflow as tf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import random
import itertools
import math
import time
import wandb
import datetime

from wandb.keras import WandbCallback
from skimage.io import imread
from skimage.transform import resize

from tensorflow import keras
# Model class
from tensorflow.keras.models import Model

# Layers
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout, Flatten, Conv2D, Dense, MaxPool2D, Conv2DTranspose, GlobalMaxPool2D, Reshape, BatchNormalization, Input
from tensorflow.keras.layers import Layer

# Optimizer
from tensorflow.keras.optimizers import Adam

# Loss function
from tensorflow.keras.losses import BinaryCrossentropy

# Data loader
from tensorflow.keras.utils import Sequence

# Metrics
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision, MSE

# ImageLoader
from skimage.io import imread

# Weights Initializer
from tensorflow.keras.initializers import RandomNormal, GlorotNormal, GlorotUniform

# Callbacks
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

wandb.init(project="ups-mv-gans", entity="charbel-abihana")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


class DataLoader(Sequence):
    
    def __init__(self, im_dir, labels= None, batch_size= 32, resize= False, output_dim= None, classification= True, shuffle= True):
        
        self.im_dir = im_dir
        self.classification = classification
        self.shuffle = shuffle
        self.resize = resize
        self.output_dim = output_dim

        self.labels = {}
        if isinstance(labels, list):
            for i, label in enumerate(labels):
                self.labels[label] = i
        elif isinstance(labels, dict):
            self.labels = labels
        else:
            for root_dir, subdirs, files in os.walk(self.im_dir):
                for i, subdir in enumerate(subdirs):
                    self.labels[subdir] = i
        
            
        self.images = glob.glob(f"{self.im_dir}/*/*")[:30_000]
        random.shuffle(self.images)
        
        if batch_size >= len(self.images):
            self.batch_size = len(self.images)
        else:
            self.batch_size = batch_size 
        self.on_epoch_end()
        
    
    def __len__(self):
        return int(np.floor(len(self.images)/self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        im_files_batch = [self.images[idx] for idx in indexes]
        if self.resize:
            batch_images = np.array([resize(imread(file), self.output_dim, preserve_range= True) for file in im_files_batch], dtype= np.uint8)
            batch_images = (np.asanyarray(batch_images, dtype= np.float32)-127.5)/127.5
        else:
            batch_images = np.arrary([imread(file) for file in im_files_batch], dtype= np.uint8)
        
        batch_labels = np.empty((self.batch_size), dtype= np.int8)
        for i, im in enumerate(im_files_batch):
            for label in self.labels.keys():
                if label in im:
                    batch_labels[i] = self.labels[label]
        return batch_images, batch_labels
    
    def on_epoch_end(self):
        "Updates all indexes after the end of each epoch"
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        
    def add_data(self, batched_data):
        return


class SaveImagesCallback(Callback):
    def __init__(self, logdir, latent_dim, save_freq):
        self.logdir = Path(f"{logdir}/gan_image_output/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.latent_dim = latent_dim
        self.save_freq = save_freq
        
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.fixed_noise = tf.random.normal([batch_size, 1, 1, self.latent_dim])
        
    def on_epoch_end(self, epoch, logs= None):
        if epoch % self.save_freq == 0:
            generator = self.model.generator
            predictions = generator(self.fixed_noise, training=False)
            
            pred_index = np.random.choice(np.array(list(range(predictions.shape[0]))), size= predictions.shape[0])
            predictions = np.array([predictions[x, :, :, :] for x in pred_index])

            plt_shape = int(np.math.sqrt(predictions.shape[0]))
            
            fig = plt.figure(figsize=(8, 8))
            for i in range(predictions.shape[0]):
                plt.subplot(plt_shape, plt_shape, i+1)
                plt.imshow(np.asarray(predictions[i, :, :, :] * 127.5 + 127.5, dtype= np.uint8))
                plt.axis('off')

            plt.savefig(f'{str(self.logdir)}/tf_image_at_epoch_{epoch:04d}.png')
            
            img_array = imread(f'{str(self.logdir)}/tf_image_at_epoch_{epoch:04d}.png')
            images = wandb.Image(img_array, caption= "Generated Images")
            wandb.log({"generated_images_example": images})


"""# Create the discriminator
discriminator = keras.Sequential(
    [
        keras.Input(shape=(128, 128, 3)),
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(64*2, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        
        layers.Conv2D(64*4, (4, 4), strides= (2, 2), padding= "same", use_bias= False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha= 0.2),

        layers.Conv2D(64*8, (4, 4), strides= (2, 2), padding= "same", use_bias= False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha= 0.2),

        layers.Conv2D(64*8, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),    
        
        layers.Conv2D(64*4, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),

        layers.Conv2D(64*2, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Conv2D(1, (3, 3), strides=(4, 4), padding="same", use_bias= False, activation= 'sigmoid'),
        
    ],
    name="discriminator",
)

# Create the generator
latent_dim = 1024
generator = keras.Sequential(
    [
        keras.Input(shape=(1, 1, latent_dim)),
        layers.Conv2DTranspose(64*12, (4, 4), strides=(1, 1), padding="valid", use_bias= False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64*8, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64*8, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64*4, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64*4, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64*2, (4, 4), strides=(2, 2), padding="same", use_bias= False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", use_bias= False, activation= "tanh"), 
    ],
    name="generator",
)"""
class Conv2DTBatchNorm(Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias):
        super(Conv2DTBatchNorm, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
    
    def build(self, input_shape):
        self.conv2d_t = Conv2DTranspose(filters= self.filters, kernel_size= self.kernel_size, strides= self.strides, use_bias= self.use_bias)
        self.batch_norm = BatchNormalization()
        self.leaky_relu = LeakyReLU()
        super(Conv2DTBatchNorm, self).build(input_shape)
    
    def call(self, input_tensor):
        x = input_tensor
        
        x = self.conv2d_t(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        
        return x  
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

class Generator(Model):
    def __init__(self, latent_dim, name, **kwargs):
        super(Generator, self).__init__(name= name, **kwargs)
        self.latent_dim = latent_dim
    
    def build(self, input_shape):
        assert input_shape[1:] == (1, 1, self.latent_dim), f"input_shape should have shape (batch_size, 1, 1, latent_dimension), received: {input_shape}"
        self.conv2d_t_1 = Conv2DTBatchNorm(filters= 64*12, kernel_size= (4, 4), strides=(1, 1), padding="valid", use_bias= False)
        self.conv2d_t_2 = Conv2DTBatchNorm(filters= 64*8, kernel_size= (2, 2), strides=(2, 2), padding="same", use_bias= False)
        self.conv2d_t_3 = Conv2DTBatchNorm(filters= 64*8, kernel_size= (2, 2), strides=(2, 2), padding="same", use_bias= False)
        self.conv2d_t_4 = Conv2DTBatchNorm(filters= 64*4, kernel_size= (2, 2), strides=(2, 2), padding="same", use_bias= False)
        self.conv2d_t_5 = Conv2DTBatchNorm(filters= 64*4, kernel_size= (2, 2), strides=(2, 2), padding="same", use_bias= False)
        self.conv2d_t_6 = Conv2DTBatchNorm(filters= 64*2, kernel_size= (2, 2), strides=(2, 2), padding="same", use_bias= False)
        
        self.conv2d = Conv2D(filters= 3, kernel_size= (3, 3), strides=(1, 1), padding="same", use_bias= False, activation= "tanh")
        super(Generator, self).build(input_shape)
    
    def call(self, input_tensor):
        x = input_tensor
        x = self.conv2d_t_1(x)
        x = self.conv2d_t_2(x)
        x = self.conv2d_t_3(x)
        x = self.conv2d_t_4(x)
        x = self.conv2d_t_5(x)
        x = self.conv2d_t_6(x)
        
        x = self.conv2d(x)
        return x
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

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim, batch_size):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        
    def call(self, data, training=False): 
        # Method needed to be implemented for tensorflow reasons when using a custom data loader
        pass
    
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = self.batch_size
        #random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_latent_vectors = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))
        
        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)
        
        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        
        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1, 1, 1)), tf.zeros((batch_size, 1, 1, 1))], axis=0
        )
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        #random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_latent_vectors = tf.random.normal(shape=(batch_size, 1, 1, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1, 1, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


# Prepare the dataset. We use both the training & test MNIST digits.
batch_size = 32
EPOCHS= 30000
EPOCH_SAVE_FREQ = 30000
LATENT_DIMENSION = 100
IMAGE_SIZE = (128, 128, 3)

train_cat_dog_data = DataLoader(im_dir= "dataset/Newdata/train_merged", resize= True, output_dim= IMAGE_SIZE, batch_size= batch_size)
data_len = len(train_cat_dog_data)

logdir = 'gan-logdir/horses_cows/'
isaveimg = SaveImagesCallback(logdir= logdir, latent_dim= LATENT_DIMENSION, save_freq= 100)
model_checkpoint = ModelCheckpoint(
                    filepath= f"{logdir}/model_checkpoint/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    monitor= "g_loss",
                    save_freq= data_len*EPOCH_SAVE_FREQ)

with tf.device('/device:GPU:1'):
    discriminator = Discriminator(name= "Discriminator")
    generator = Generator(latent_dim= LATENT_DIMENSION, name= "Generator")
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIMENSION, batch_size= batch_size)
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
    
    optimizer_d = keras.optimizers.Adam(0.0002, 0.5)
    optimizer_g = keras.optimizers.Adam(0.0002, 0.5)

    wandb.config = {
        "learning_rate": lr_schedule,
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "latent_dim": LATENT_DIMENSION,
        "image_size": IMAGE_SIZE
    }
    gan.compile(
        #d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        #g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        d_optimizer= optimizer_d,
        g_optimizer= optimizer_g,
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    gan.fit(train_cat_dog_data, epochs=EPOCHS, callbacks= [isaveimg, WandbCallback()], workers= 16)

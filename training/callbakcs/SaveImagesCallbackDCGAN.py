import datetime

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback

class SaveImagesCallbackDCGAN(Callback):
    def __init__(self, logdir, latent_dim, save_freq, batch_size):
        self.logdir = Path(f"{logdir}/gan_image_output/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.latent_dim = latent_dim
        self.save_freq = save_freq
        
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        
        self.fixed_noise = tf.random.normal([self.batch_size, 1, 1, self.latent_dim])
        
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
            plt.show()
            plt.savefig(f'{str(self.logdir)}/tf_image_at_epoch_{epoch:04d}.png')
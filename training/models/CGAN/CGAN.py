import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from ..CGAN import *

class CGAN():
    def __init__(self, image_size, num_classes, latent_dim, generator, discriminator):
        # Input shape
        self.img_rows, self.cols, self.channels = image_size
        self.img_shape = image_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = build_discriminator(self.img_shape, self.num_classes)
        
        #self.discriminator.build(input_shape= [(None, *self.img_shape), (None, 1,)])
        
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = build_generator(latent_dim= self.latent_dim, num_classes= self.num_classes)
        #self.generator.build(input_shape= [(None, self.latent_dim), (None, 1,)])

        # The generator takes noise and the target label as input
        # and generates the corresponding class of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def train(self, epochs, dataset, batch_size=128, sample_interval=50):
        D_loss, G_loss, acc = [], [], []
        # Load iterator dataset assuming DataLoader with one batch (full data)
        X_train, y_train = next(iter(dataset))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            D_loss.append(d_loss[0])
            G_loss.append(g_loss)
            acc.append(100*d_loss[1])
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([[0], [0], [0], [0], [1], [1], [1], [1]]).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = gen_imgs

        fig, axs = plt.subplots(r, c)
        for i in range(r):
            if i == 0:
                cnt = 0
                title= "Cow"
            elif i == 1:
                cnt = 4
                title= "Horse"
            for j in range(c):
                axs[i, j].imshow(np.asarray(gen_imgs[cnt,:,:, :]* 127.5 + 127.5, dtype= np.uint8))
                axs[i, j].set_title(title)
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
        plt.close()
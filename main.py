import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb

from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MSE, Recall, Precision, BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy

from training.data_pre_processing import DataLoader
from training.models.classification import Classifier
from training.models.CGAN import CDiscriminator, CGenerator, CGAN
from training.models.DCGAN import Generator, Discriminator, DCGAN
from training.models.CycleGAN import CycleGAN

from training.callbakcs import SaveImagesCallbackDCGAN, CycleGANMonitor

from training.utils.metrics import *
#wandb.init()

# To limit GPU VRAM allocation by tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


train_horses_cows = DataLoader(im_dir= "dataset/Newdata/Train", classification= True, output_dim= (128, 128, 3), batch_size= 16)
val_horses_cows = DataLoader(im_dir= "dataset/Newdata/Test", classification= True, output_dim= (128, 128, 3))

# Get a batch of data
horses_cows_iter = iter(train_horses_cows)
batch_data = next(horses_cows_iter)

horse_batch = [data for i, data in enumerate(batch_data[0]) if batch_data[1][i] == 1]
cow_batch = [data for i, data in enumerate(batch_data[0]) if batch_data[1][i] == 0]

random_horse = np.random.choice(range(len(horse_batch)), size= 4, replace= False)
random_cow = np.random.choice(range(len(cow_batch)-1), size= 4, replace= False)

cow_images, horse_images = [cow_batch[i] for i in random_cow], [horse_batch[i] for i in random_horse]
fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, [*cow_images, *horse_images]):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.show()

EPOCHS= 50
BATCH_SIZE= 16
IMAGE_SIZE= (128, 128, 3)

train_horses_cows_classification = DataLoader(im_dir= "dataset/Newdata/Train", classification= True, output_dim= IMAGE_SIZE, batch_size= BATCH_SIZE)
val_horses_cows_classification = DataLoader(im_dir= "dataset/Newdata/Test", classification= True, output_dim= IMAGE_SIZE, batch_size= BATCH_SIZE)

horses_cows_classifier = Classifier(name= "Horses_vs_Cows_Classifier")
horses_cows_classifier.build(input_shape= (None, *IMAGE_SIZE))

# TODO Setup WANDB for classification logging
# wandb run name
# wandb log

horses_cows_classifier.summary()
horses_cows_classifier.compile(optimizer= Adam(0.001), loss= BinaryCrossentropy(), metrics= [BinaryAccuracy(), Recall(), Precision(), MSE])
model_data = horses_cows_classifier.fit(train_horses_cows_classification, validation_data= val_horses_cows_classification, epochs= EPOCHS, workers= 10)

show_metrics_classification(horses_cows_classifier.history)


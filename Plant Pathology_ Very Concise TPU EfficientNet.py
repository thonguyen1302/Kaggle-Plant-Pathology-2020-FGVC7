#!/usr/bin/env python
# coding: utf-8

# ## About this kernel
# 
# In my last [TPU kernel for the flower competition](https://www.kaggle.com/xhlulu/flowers-tpu-concise-efficientnet-b7), I wrapped the very [comprehensive starter kernel](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu) to show how to load `TFRecords` in order to predict flower categories.
# 
# In this kernel, I want to show the simplest and most barebone way to load `png` files (instead of `TFRecords`). In here, I only included the commands you will need to train the model; no bells and whistles included, which means there are no util functions to display the images or preprocess the images, but just enough content for you to quickly understand how `tf.data.Dataset` works.
# 
# If you want to dive deeper in the `tf.data.Dataset` way of building your input pipeline, please check out [this tutorial by Martin](https://codelabs.developers.google.com/codelabs/keras-flowers-data/#0), which I followed in order to build this kernel.
# 
# ### References
# 
# * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
# * https://codelabs.developers.google.com/codelabs/keras-flowers-data/#0

# In[1]:


import tensorflow as tf
tf.test.is_gpu_available()


# In[2]:


import math, re, os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split


# ## TPU Config

# In[3]:


AUTO = tf.data.experimental.AUTOTUNE

# Create strategy from tpu
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# strategy = tf.distribute.experimental.TPUStrategy(tpu)

# Data access
# GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 50
BATCH_SIZE = 8
IMAGE_SIZE = 256


# ## Load label and paths

# In[4]:


def format_path(st):
    return 'data/images/' + st + '.jpg'


# In[5]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')
print(train.shape)
print(test.shape)
train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train.loc[:, 'healthy':].values

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020)


# ## Create Dataset objects
# 
# A `tf.data.Dataset` object is needed in order to run the model smoothly on the TPUs. Here, I heavily trim down [my previous kernel](https://www.kaggle.com/xhlulu/flowers-tpu-concise-efficientnet-b7), which was inspired by [Martin's kernel](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu).

# In[6]:


def decode_image(filename, label=None, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label


# In[7]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# ## Modelling

# ### Helper Functions

# In[8]:


def build_lrfn(lr_start=0.00001, lr_max=0.00003, 
               lr_min=0.00001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * 2

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


# ### Load Model into TPU

# In[9]:


# with strategy.scope():
model = tf.keras.Sequential([
    efn.EfficientNetB7(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights='imagenet',
        include_top=False
    ),
    L.GlobalAveragePooling2D(),
    L.Dense(train_labels.shape[1], activation='softmax')
])
model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['categorical_accuracy']
)
model.summary()


# ### Start training

# In[ ]:


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

history = model.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset
)





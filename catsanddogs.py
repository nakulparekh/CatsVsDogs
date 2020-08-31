# NOTE: THIS WAS RAN USING GOOGLE COLAB, THEREFORE ALL DIRECTORIES WERE MADE USING
# GOOGLE DRIVE DIRECTORIES

####################################
# IMPORTS                          #
####################################
# web scraper
import requests
import json
import urllib.request
import random
from bs4 import BeautifulSoup
from google.colab import drive

# seed
import tensorflow
from numpy.random import seed

# sample pyplots
from matplotlib import pyplot
from matplotlib.image import imread

# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from keras.preprocessing.image import load_img, img_to_array

# model training
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
import keras

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# organize dataset into a useful structure
import os
import shutil
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# metrics
from sklearn import metrics
import pandas as pd

####################################
# SETUP                            #
####################################
# directory
drive.mount('/content/drive')
data_dir = 'drive/My Drive/Colab Notebooks/Web scraper/dataset_dogs_vs_cats/'

#setting seed for all random numbers
seed(1)
tensorflow.random.set_seed(2)

####################################
# WEB SCRAPER                      #
####################################
#COMMENT OUT WHEN NOT IN USE
def fetchImages(base_url, maximum, res):
  url_list = []                                                               # creates empty list that will contain the urls
  chunk_size = 30                                                             # amount of photos per page

  def fetchChunk(idx):                                                        # fetch images on a given page index using requests
    url = '%s?page=%d&per_page=%d' % (base_url, idx, chunk_size)
    return requests.get(url).text

  def parseChunk(chunk):                                                      # parse the received chunk from a string to a dictionary
    data = json.loads(chunk)                                                  # json library does parsing
    images = data['photos']                                                   # 'photos' is the sub-dictionnary containing the images

    for img in images:                                                        # iterate through each photo from page gets url
      img_url = img['urls'][res]                                              # returns 5 urls, one for each resolution
      url_list.append(img_url)                                                # appends url to the list

  idx = 0                                                                     # current page index
  while len(url_list) < maximum:                                              # fetches pages until less than the max amount is acquired
    chunk = fetchChunk(idx)                                                   # fetches the chunk
    parseChunk(chunk)                                                         # parses the chunk
    idx += 1                                                                  # increment index by 1

  url_list = url_list[:maximum]                                               # trims list to contain max urls wanted
  return url_list

animal = 'dog'                                                                # change the ending of url for other dataset
base = 'https://unsplash.com/napi/landing_pages/images/animals/'              # base of url
url = base + animal                                                           # concatenates ending of url to base of url for other dataset

resolution = 'regular'                                                        # just chooses normal resolution
maximum = 5000                                                                # desired number of pictures

picture_array = fetchImages(url, maximum, resolution)                         # array of all urls

for i in range(len(picture_array)):
  img_name = i+1                                                              # creates a unique name
  full_name = data_dir + animal + str(img_name) + ".jpg"                      # adds file type name
  urllib.request.urlretrieve(picture_array[i], full_name)                     # fetch image of url and save into dir

print("scraping complete")

####################################
# LOADING DATA                     #
####################################
# define location of dataset
folder = 'drive/My Drive/Colab Notebooks/Web scraper/dataset_dogs_vs_cats/raw/'
photos, labels = list(), list()
subdirectories = ['cats/','dogs/']

for sub in subdirectories:
	direct = folder + sub
	for file in listdir(direct): 												# enumerate files in the directory
		output = 0.0															# determine class
		if file.startswith('cat'):
			output = 1.0
		photo = load_img(direct + file, target_size=(125, 125))					# load image
		photo = img_to_array(photo)												# convert to numpy array
		photos.append(photo)													# store
		labels.append(output)

photos = asarray(photos)														# convert to a numpy arrays
labels = asarray(labels)
print(photos.shape, labels.shape)

np.save(data_dir + 'dogs_vs_cats_photos.npy', photos)                           # save the reshaped photos
np.save(data_dir + 'dogs_vs_cats_labels.npy', labels)

photos = np.load(data_dir + 'dogs_vs_cats_photos.npy')                          # load the reshaped photos
labels = np.load(data_dir + 'dogs_vs_cats_labels.npy')

#NOTE ONLY NEED TO RUN ONCE TO SACE DATASET, THEN ONLY NEED TO RUN 2 LINES TO LOAD

####################################
# split photos and labels into train, test, and validation sets
photos_train, photos_test, labels_train, labels_test = train_test_split (photos, labels, test_size=0.2, random_state=30)

photos_test, photos_val, labels_test, labels_val = train_test_split(photos_test, labels_test, test_size=0.5, random_state=30)

# rescale train, val and test
train_datagen = ImageDataGenerator(rescale = 1.0/255.0)
photos_train_resize = train_datagen.flow(
    photos_train,
    y = None,
    batch_size=1,
    shuffle=False
)

val_datagen = ImageDataGenerator(rescale = 1.0/255.0)
photos_val_resize = val_datagen.flow(
    photos_val,
    y = None,
    batch_size=1,
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
photos_test_resize = test_datagen.flow(
    photos_test,
    y = None,
    batch_size=1,
    shuffle=False
)

# extract all elems without reshape
photos_resize_train = [photos_train_resize.next() for i in range(len(photos_train))]
photos_resize_train = np.array(photos_resize_train).reshape((8000, 125, 125, 3))

photos_resize_val = [photos_val_resize.next() for i in range(len(photos_val))]
photos_resize_val = np.array(photos_resize_val).reshape((1000, 125, 125, 3))

photos_resize_test = [photos_test_resize.next() for i in range(len(photos_test))]
photos_resize_test = np.array(photos_resize_test).reshape((1000, 125, 125, 3))

####################################
# Defining functinos               #
####################################
def define_model():
  model = Sequential()                                                          # used for a plain stack of layers
  model.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(125,125,3)))
  model.add(Conv2D(32,(3,3), activation='relu', padding='same'))

  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.125))

  model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
  model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
  model.add(Conv2D(64,(3,3), activation='relu', padding='same'))

  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.125))

  model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
  model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
  model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
  model.add(Conv2D(128,(3,3), activation='relu', padding='same'))

  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.125))
  model.add(Flatten())

  model.add(Dense(200, activation='relu',
                  kernel_regularizer=keras.regularizers.l1()))
  model.add(Dense(100, activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.001)))
  model.add(Dense(50, activation='relu',
                  kernel_regularizer=keras.regularizers.l1_l2(0.01)))
  model.add(Dense(10, activation='relu'))

  model.add(Dense(1, activation='sigmoid'))

  opt = Adam(learning_rate=0.0001)                                            # compile model
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

  return model

def run_metrics():
    model.summary()

    results = model.evaluate(photos_test, labels_test, batch_size=64)
    print(results[1]*100.0)

    y_pred = model.predict(photos_test)
    thres = 0.5
    y_pred_new = [(0 if y_pred[i] < thres else 1) for i in range (len(y_pred))]

    fpr, tpr, thresholds= metrics.roc_curve(labels_test, y_pred_new, pos_label =1)
    roc_auc = metrics.auc(fpr, tpr)
    fig = plt.figure(figsize=(12,8))
    plt.plot (fpr, tpr, 'c', label ='AUC = %0.2f' %roc_auc)
    plt.legend (loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    # fig.savefig(data_dir + 'auc.jpg')

    # Print f1, precision, and recall scores
    precision = metrics.precision_score(labels_test, y_pred_new)
    confusion_matrix = metrics.confusion_matrix(labels_test, y_pred_new)
    recall = metrics.recall_score(labels_test, y_pred_new)
    f1_score = metrics.f1_score(labels_test, y_pred_new)
    kappa = metrics.cohen_kappa_score(labels_test, y_pred_new)
    accuracy = metrics.accuracy_score (labels_test, y_pred_new)

    print("Confusion matrix:")
    print(pd.DataFrame (confusion_matrix, columns = ['pred_cat', 'pred_dog'], index = ['cat', 'dog']))

    print("\nPrecision: {}%".format (round(precision*100, 5)))
    print("Recall: {}%".format(round (recall*100, 5)))
    print("F1 Score: {}%".format (round(f1_score*100, 5)))
    print("Kappa Score: {}%".format (round(kappa*100, 5)))
    print("Accuracy: {:.5f}%".format(round(accuracy*100,5)))

    # pyplots (can only be ran if model is trained, not loaded)
    print(history.history.keys())

    fig1 = plt.figure(figsize=(12,8))                                           # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # fig1.savefig(data_dir + '/accuracyw.jpg')

    fig2 = plt.figure(figsize=(12,8))                                           # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # fig2.savefig(data_dir + '/lossw.jpg')

####################################
# Main                             #
####################################
model = define_model()
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(photos_train, labels_train,
                    validation_data = (photos_val, labels_val),
                    batch_size = 64, epochs = 200, callbacks=[es])

# saving model
model.save(data_dir + 'my_model_one.h5')

# loading model
model = keras.models.load_model(data_dir + 'my_model_one.h5')

run_metrics()

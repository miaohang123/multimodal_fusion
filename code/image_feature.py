import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from helper import data_iterator

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
set_session(tf.Session(config=config))

image_width = 150
image_height = 150

class ImageFeature(object):
    def __init__(self, datapath):
        self.image_path_list = data_iterator.DataIterator('imdb', datapath, 'model').image_path_list
        self.model()

    def model(self):
        self.model = InceptionV3(weights='imagenet', include_top=False)
        self.model = VGG16(weights='imagenet', include_top=False)
        #self.model = VGG19(weights='imagenet', include_top=False)
        #print(self.model.summary())


    def extract_feature(self, filepath):
        img = image.load_img(filepath, target_size=(image_width, image_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features

    def save_feature(self, save_path):
        res = []
        for image_path in self.image_path_list:
            image_feature = self.extract_feature(image_path[3:])
            res.append(image_feature)
        res = np.array(res)
        np.save(save_path, res)


if __name__ == '__main__':
    image_feature_train = ImageFeature(datapath='../imdb/valid_data_path/multimodal.train.txt')
    #print(image_feature_object.image_path_list)
    image_feature_train.save_feature(save_path='../imdb/vgg16_feature/train.npy')

    image_feature_val = ImageFeature(datapath='../imdb/valid_data_path/multimodal.val.txt')
    image_feature_val.save_feature(save_path='../imdb/vgg16_feature/val.npy')

    image_feature_test = ImageFeature(datapath='../imdb/valid_data_path/multimodal.test.txt')
    image_feature_test.save_feature(save_path='../imdb/vgg16_feature/test.npy')

    #
    # feature_vector = image_feature_object.extract_feature(filepath='../reddit/new_image/train/happy/1a3n9s.png')
    #
    # print(feature_vector[0].shape)
    # print(feature_vector[0])


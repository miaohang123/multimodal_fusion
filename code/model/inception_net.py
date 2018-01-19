import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

class ICNNConfig(object):
    num_classes = 4  # 类别数
    hidden_dim = 1024

    batch_size = 64  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

class DataGenerator(object):
    def __init__(self, train_dir, val_dir, test_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.getTrainGen()
        self.getValGen()
        self.getTestGen()

    def getTrainGen(self):
        train_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

    def getValGen(self):
        val_datagen = image.ImageDataGenerator(rescale=1. / 255)

        self.val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

    def getTestGen(self):
        test_datagen = image.ImageDataGenerator(rescale=1. / 255)

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

class InceptionNet(object):
    def __init__(self, config):
        self.config = config
        self.model = self.model()

    def model(self):
        # create the base pre-trained model
        self.base_model = InceptionV3(weights='imagenet', include_top=False)
        print(self.base_model.summary())
        #global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        #fully-connected layer
        x = Dense(self.config.hidden_dim, activation='relu')(x)
        #logistic layer
        predictions = Dense(self.config.num_classes, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        #freeze all convolutional InceptionV3 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        #print(self.model.summary())
        #self.train()

    def fine_tuning(self):
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def train(self):
        dgen = DataGenerator(train_dir='../reddit/new_image/train',
                             val_dir='../reddit/new_image/val',
                             test_dir='../reddit/new_image/test')
        train_generator = dgen.train_generator
        val_generator = dgen.val_generator
        test_generator = dgen.test_generator
        self.model.fit_generator(train_generator,
            steps_per_epoch=2575,
            epochs=50,
            validation_data=val_generator,
            validation_steps=321)

    def predict(self):
        pass

if __name__ == '__main__':
    config = ICNNConfig()
    model = InceptionNet(config)
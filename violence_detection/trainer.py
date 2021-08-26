import numpy as np
import random
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow

# tensorflow imports
from tensorflow.keras import Sequential, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# # # Creating Train / Val / Test folders

# root_dir = '../raw_data/frames_dataset' # data root path
# classes_dir = ['violence', 'non_violence'] #total labels

# val_ratio = 0.15
# test_ratio = 0.05

# for cls in classes_dir:
#     os.makedirs(root_dir +'/train/' + cls)
#     os.makedirs(root_dir +'/val/' + cls)
#     os.makedirs(root_dir +'/test/' + cls)


#     # Creating partitions of the data after shuffling


#     src = root_dir + "/" + cls # Folder to copy images from

#     allFileNames = os.listdir(src)
#     np.random.shuffle(allFileNames)
#     train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
#                                                           [int(len(allFileNames)* (1 - (val_ratio + test_ratio))),
#                                                            int(len(allFileNames)* (1 - test_ratio))])


#     train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#     val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
#     test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

#     # Copy-pasting images
#     for name in train_FileNames:
#         shutil.copy(name, root_dir +'/train/' + cls)

#     for name in val_FileNames:
#         shutil.copy(name, root_dir +'/val/' + cls)

#     for name in test_FileNames:
#         shutil.copy(name, root_dir +'/test/' + cls)

# print('Total images: ', len(allFileNames))
# print('Training: ', (len(train_FileNames)/len(allFileNames))*100)
# print('Validation: ', (len(val_FileNames)/len(allFileNames))*100)
# print('Testing: ', (len(test_FileNames)/len(allFileNames))*100)

# # import data
# root = r'../raw_data/frames_dataset'

# train_dir = root + "/train/"

# train_datagen = ImageDataGenerator(rescale=1.0/255.,
#                                       rotation_range=40,
#                                       width_shift_range=0.2,
#                                       height_shift_range=0.2,
#                                       shear_range=0.2,
#                                       zoom_range=0.2,
#                                       horizontal_flip=True,
#                                       fill_mode='nearest'
#                                     )

# train_generator = train_datagen.flow_from_directory(train_dir,
#                                                     batch_size=16,
#                                                     classes=['violence',
#                                                              'non_violence'
#                                                              ],
#                                                     class_mode="categorical",
#                                                     color_mode="rgb",
#                                                     target_size=(224, 224))

# val_dir = root + "/val/"
# validation_datagen = ImageDataGenerator(rescale=1.0/255.)

# validation_generator = validation_datagen.flow_from_directory(val_dir,
#                                                               batch_size=16,
#                                                                 classes=['violence',
#                                                                          'non_violence'
#                                                                         ],
#                                                               class_mode="categorical",
#                                                               color_mode="rgb",
#                                                               target_size=(224, 224)
#                                                               )

# def load_vgg19(dense_n=512, lr= 0.000134,):
#     transfer_model = tensorflow.keras.applications.VGG19(
#                                             include_top=False, weights="imagenet",
#                                             input_shape=(224, 224, 3), pooling="max", classes=2,
#     )
#     transfer_model.trainable = False
#     model = tensorflow.keras.models.Sequential([
#                             transfer_model,
#                             tensorflow.keras.layers.Flatten(),
#                             tensorflow.keras.layers.Dense(dense_n, activation='relu'),
#                             tensorflow.keras.layers.Dense(1, activation='sigmoid')
#                                 ])

#     opt = optimizers.Adam(learning_rate=lr)
#     model.compile(loss='binary_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model

# model = load_vgg19()

# es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20, verbose=1, restore_best_weights=True)

# model.fit(train_generator,
#                     validation_data=validation_generator,
#                     epochs= 5,
#                     verbose=1,
#                     callbacks = es)


# Build a trainer class
class Trainer():
    def __init__(self):
        self.root = '../raw_data/frames_dataset'


# Splitting frames into train, val, test
    def split(self, root_dir):
        self.val_ratio = 0.15
        self.test_ratio = 0.05

        root_dir = self.root # data root path
        classes_dir = ['violence', 'non_violence'] # all labels

        for label in classes_dir:
            os.makedirs(root_dir + '/train/' + label)
            os.makedirs(root_dir + '/val/' + label)
            os.makedirs(root_dir + '/test/' + label)


            # Creating partitions of the data after shuffling
            src = root_dir + "/" + label  # Folder to copy images from

            allFileNames = os.listdir(src)
            np.random.shuffle(allFileNames)
            train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames)* (1 - (val_ratio + test_ratio))),
                                                                  int(len(allFileNames)* (1 - test_ratio))])


            train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
            val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
            test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

            # Copy-pasting images
            for name in train_FileNames:
                shutil.copy(name, root_dir + '/train/' + label)

            for name in val_FileNames:
                shutil.copy(name, root_dir + '/val/' + label)

            for name in test_FileNames:
                shutil.copy(name, root_dir + '/test/' + label)


# Generate data + Augment frames in the train set
    def generate_data(self):
        self.split()
        train_dir = self.root + "/train/"

        train_datagen = ImageDataGenerator(rescale=1.0/255.,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest'
                                    )

        self.train_generator = train_datagen.flow_from_directory(train_dir,
                                                            batch_size=16,
                                                            classes=['violence',
                                                                    'non_violence'
                                                                    ],
                                                            class_mode="categorical",
                                                            color_mode="rgb",
                                                            target_size=(224, 224))

        val_dir = self.root + "/val/"
        validation_datagen = ImageDataGenerator(rescale=1.0/255.)

        self.validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                                      batch_size=16,
                                                                        classes=['violence',
                                                                                'non_violence'
                                                                                ],
                                                                      class_mode="categorical",
                                                                      color_mode="rgb",
                                                                      target_size=(224, 224))
        return self.train_generator, self.validation_generator

# Transfer learning model
    def load_vgg19(self, dense_n=512, lr= 0.000134,):
        transfer_model = tensorflow.keras.applications.VGG19(
                                                include_top=False, weights="imagenet",
                                                input_shape=(224, 224, 3), pooling="max", classes=2,
        )
        transfer_model.trainable = False
        self.model = tensorflow.keras.models.Sequential([
                                transfer_model,
                                tensorflow.keras.layers.Flatten(),
                                tensorflow.keras.layers.Dense(dense_n, activation='relu'),
                                tensorflow.keras.layers.Dense(1, activation='sigmoid')
                                    ])

        opt = optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return self.model

# Instantiate  + fit model
    def run(self):
        self.model = load_vgg19()
        es = EarlyStopping(monitor='val_accuracy',
                           mode='max',
                           patience=20,
                           verbose=1,
                           restore_best_weights=True)

        self.model.fit(train_generator,
                  validation_data=validation_generator,
                  epochs=5,
                  verbose=1,
                  callbacks=es)

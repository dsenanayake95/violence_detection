import numpy as np
import random
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow
import cropper

# tensorflow imports
from tensorflow.keras import Sequential, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ZIP = "raw_data/violence.zip"  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CHANGE
CWD_PATH = os.getcwd()
ROOT = os.path.join(CWD_PATH, "raw_data/cropped_dataset/")  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CHANGE
# BUCKET_NAME = "YOURBUCKETNAME"  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CHANGE

# TARGET_IMSIZE = (224,224)

# Build a trainer class
class Trainer():
    def __init__(self):
        self.train_flow = None
        self.val_flow = None
        self.model = None
        self.history = None
        self.eval_ = None

    # def unzipper(self):
    #     with zipfile.ZipFile(ZIP, 'r') as zip_ref:
    #         zip_ref.extractall(".")
    #     return self

    # Splitting frames into train, val, test
    def split(self):
        self.val_ratio = 0.15
        self.test_ratio = 0.05

        self.root = ROOT # data root path
        classes_dir = ['violence', 'non_violence'] # all labels

        for label in classes_dir:
            os.makedirs(ROOT + '/train/' + label)
            os.makedirs(ROOT + '/val/' + label)
            os.makedirs(ROOT + '/test/' + label)

            # Creating partitions of the data after shuffling
            src = ROOT + "/" + label  # Folder to copy images from

            self.allFileNames = os.listdir(src)
            np.random.shuffle(self.allFileNames)
            self.train_FileNames, self.val_FileNames, self.test_FileNames = np.split(np.array(self.allFileNames),
                                                                  [int(len(self.allFileNames)* (1 - (self.val_ratio + self.test_ratio))),
                                                                  int(len(self.allFileNames)* (1 - self.test_ratio))])

            self.train_FileNames = [src+'/'+ name for name in self.train_FileNames.tolist()]
            self.val_FileNames = [src+'/' + name for name in self.val_FileNames.tolist()]
            self.test_FileNames = [src+'/' + name for name in self.test_FileNames.tolist()]

            # Copy-pasting images
            for name in self.train_FileNames:
                shutil.copy(name, ROOT + '/train/' + label)

            for name in self.val_FileNames:
                shutil.copy(name, ROOT + '/val/' + label)

            for name in self.test_FileNames:
                shutil.copy(name, ROOT + '/test/' + label)


    # Generate data + Augment frames in the train set
    def generate_data(self):
        train_dir = ROOT + "/train/"

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

        val_dir = ROOT + "/val/"
        validation_datagen = ImageDataGenerator(rescale=1.0/255.)

        self.validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                                      batch_size=16,
                                                                        classes=['violence',
                                                                                'non_violence'
                                                                                ],
                                                                      class_mode="categorical",
                                                                      color_mode="rgb",
                                                                      target_size=(224, 224))
        return self

    # Transfer learning model
    def model(self, dense_n=512, lr= 0.000134,):
        transfer_model = tensorflow.keras.applications.VGG19(
                                                include_top=False, weights="imagenet",
                                                input_shape=(224, 224, 3), pooling="max", classes=2,
        )
        transfer_model.trainable = False
        self.model = tensorflow.keras.models.Sequential([
                                transfer_model,
                                tensorflow.keras.layers.Flatten(),
                                tensorflow.keras.layers.Dense(dense_n, activation='relu'),
                                tensorflow.keras.layers.Dense(dense_n/2, activation='relu'),
                                tensorflow.keras.layers.Dense(dense_n/4, activation='relu'),
                                tensorflow.keras.layers.Dense(1, activation='sigmoid')
                                    ])

        opt = optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return self.model

    # Instantiate  + fit model
    def run(self):
        self.model = self.load_vgg19()
        es = EarlyStopping(monitor='val_accuracy',
                           mode='max',
                           patience=20,
                           verbose=1,
                           restore_best_weights=True)

        self.model.fit(self.train_generator,
                  validation_data=self.validation_generator,
                  epochs=5,
                  verbose=1,
                  callbacks=es)


if __name__ == "__main__":
    print("loading trainer...")
    trainer = Trainer(root='../raw_data/frames_dataset')
    print("trainer loaded")
    print("spliting the data into train, val, test...")
    trainer.split()
    print("data successfully split")
    print('Total images: ', len(trainer.allFileNames))
    print('Training: ', len(trainer.train_FileNames))
    print('Validation: ', len(trainer.val_FileNames))
    print('Testing: ', len(trainer.test_FileNames))

    print("augmenting data now...")
    trainer.generate_data()
    print("data successfully augmented")
    print("loading vgg19...")
    trainer.model()
    print("vgg19 successfully loaded")
    print("running model...")
    trainer.run()
    print("model completed")
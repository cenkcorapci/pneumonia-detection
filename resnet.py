import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
import keras
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras.callbacks import *

from livelossplot import PlotLossesKeras
from matplotlib import pyplot as plt
from data_generator import *
from config import *


class ResNetModel:
    def __init__(self, nb_epochs=20, early_stopping_patience=4, n_valid_samples=2560, debug_sample_size=None):
        self.model_name = 'resnet'
        self.weight_file_path = MODEL_BINARIES_PATH + self.model_name + '.h5'
        self.n_valid_samples = n_valid_samples
        self.nb_epochs = nb_epochs

        learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(self.cosine_annealing)

        tb_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=0, write_graph=True,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        self.callbacks = [learning_rate_callback, tb_callback, PlotLossesKeras()]
        self.callbacks.append(EarlyStopping(monitor='loss', patience=early_stopping_patience))
        self.callbacks.append(ReduceLROnPlateau(monitor='loss',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001))
        if debug_sample_size is not None:
            self.debug_sample_size = debug_sample_size
        self.load_data()
        self.model = self.compile_network()

    def create_downsample(self, channels, inputs):
        x = keras.layers.BatchNormalization(momentum=0.99999)(inputs)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
        x = keras.layers.MaxPool2D((2, 2))(x)
        return x

    def create_resblock(self, channels, inputs):
        x = keras.layers.BatchNormalization(momentum=0.9999)(inputs)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization(momentum=0.9999)(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
        return keras.layers.add([x, inputs])

    def create_network(self, input_size, channels, n_blocks=2, depth=4):
        # input
        inputs = keras.Input(shape=(input_size, input_size, 1))
        x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
        # residual blocks
        for d in range(depth):
            channels = channels * 2
            x = self.create_downsample(channels, x)
            for b in range(n_blocks):
                x = self.create_resblock(channels, x)
        # output
        x = keras.layers.BatchNormalization(momentum=0.9999)(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        outputs = keras.layers.UpSampling2D(2 ** depth)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_network(self):
        # create network and compiler
        model = self.create_network(input_size=256, channels=32, n_blocks=2, depth=4)
        model.compile(optimizer='adam',
                      loss=self.iou_bce_loss,
                      metrics=['accuracy', self.mean_iou])
        return model

    # define iou or jaccard loss function
    def iou_loss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
        return 1 - score

    # combine bce loss and iou loss
    def iou_bce_loss(self, y_true, y_pred):
        return 0.4 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.6 * self.iou_loss(y_true, y_pred)

    # mean iou as a metric
    def mean_iou(self, y_true, y_pred):
        y_pred = tf.round(y_pred)
        intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        smooth = tf.ones(tf.shape(intersect))
        return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

    # cosine learning rate annealing
    def cosine_annealing(self, x):
        lr = 0.01
        epochs = 20
        return lr * (np.cos(np.pi * x / epochs) + 1.) / 2

    def load_data(self):
        # load and shuffle filenames
        filenames = os.listdir(TRAIN_IMAGES_PATH)
        random.shuffle(filenames)
        self.pneumonia_locations = get_pneumonia_locations()
        # split into train and validation filenames
        try:
            filenames = filenames[:self.debug_sample_size]
            self.n_valid_samples = int(self.debug_sample_size / 10)
        except Exception:
            logging.warning("Using the complete data set.")
        self.train_filenames = filenames[self.n_valid_samples:]
        self.valid_filenames = filenames[:self.n_valid_samples]
        logging.info('n train samples', len(self.train_filenames))
        logging.info('n valid samples', len(self.valid_filenames))
        n_train_samples = len(filenames) - self.n_valid_samples
        logging.info("Loaded data, {0} training samples.".format(n_train_samples))

    def load_model(self):
        if not os.path.exists(self.weight_file_path):
            logging.info("Can not find a pretrained {0} weights on s3, training a new one...".format(self.model_name))
            self.train()
        else:
            self.model.load_weights(self.weight_file_path)
            self.model.compile(optimizer='adam', loss=self.iou_bce_loss, metrics=['accuracy', self.mean_iou])
        return self

    def train(self):
        # create train and validation generators
        train_gen = generator(TRAIN_IMAGES_PATH, self.train_filenames, self.pneumonia_locations, batch_size=16, image_size=256,
                              shuffle=True,
                              augment=True, predict=False)
        valid_gen = generator(TRAIN_IMAGES_PATH, self.valid_filenames, self.pneumonia_locations, batch_size=16, image_size=256,
                              shuffle=False,
                              predict=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _ = self.model.fit_generator(
                train_gen,
                validation_data=valid_gen,
                verbose=1,
                callbacks=self.callbacks,
                epochs=self.nb_epochs,
                shuffle=True)

            logging.info("Training complete")
            self.model.save_weights(self.weight_file_path)
            logging.info("Saved model to disk")

    def generate_submission(self):
        # load and shuffle filenames
        test_filenames = os.listdir(TEST_IMAGES_PATH)

        try:
            test_filenames = test_filenames[:int(self.debug_sample_size / 10)]
            logging.warning("This submission file is incomplete for debug purpose.")
        except Exception:
            logging.info('n test samples:', len(test_filenames))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # create test generator with predict flag set to True
            test_gen = generator(TEST_IMAGES_PATH,
                                 test_filenames,
                                 None,
                                 batch_size=20,
                                 image_size=256,
                                 shuffle=False,
                                 predict=True)

            logging.info("Generating submission...")
            # create submission dictionary
            submission_dict = {}
            # loop through testset
            for imgs, filenames in test_gen:
                # predict batch of images
                preds = self.model.predict(imgs)
                # loop through batch
                for pred, filename in zip(preds, filenames):
                    # resize predicted mask
                    pred = resize(pred, (1024, 1024), mode='reflect')
                    # threshold predicted mask
                    comp = pred[:, :, 0] > 0.5
                    # apply connected components
                    comp = measure.label(comp)
                    # apply bounding boxes
                    predictionString = ''
                    for region in measure.regionprops(comp):
                        # retrieve x, y, height and width
                        y, x, y2, x2 = region.bbox
                        height = y2 - y
                        width = x2 - x
                        # proxy for confidence score
                        conf = np.mean(pred[y:y + height, x:x + width])
                        # add to predictionString
                        predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(
                            height) + ' '
                    # add filename and predictionString to dictionary
                    filename = filename.split('.')[0]
                    submission_dict[filename] = predictionString
                # stop if we've got them all
                if len(submission_dict) >= len(test_filenames):
                    break

            # save dictionary as csv file
            logging.info("Persisting submission...")
            sub = pd.DataFrame.from_dict(submission_dict, orient='index')
            sub.index.names = ['patientId']
            sub.columns = ['PredictionString']
            sub.to_csv(SUBMISSIONS_FOLDER_PATH + self.model_name + '_submission.csv')


if __name__ == "__main__":
    resnet = ResNetModel(nb_epochs=20)
    resnet.train()
    resnet.generate_submission()

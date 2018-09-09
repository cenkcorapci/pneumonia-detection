import tensorflow as tf
from keras.callbacks import *
from livelossplot import PlotLossesKeras
from skimage import measure
import keras
from keras.layers import *
from config import *
from data_generator import *
from ai_utils import *
from keras.applications import *


class EncoderModelType:
    resnet = "ResNet"
    vgg16 = "vgg16"


class EncoderDecoderModel:
    def __init__(self, encoder_model=EncoderModelType.resnet, nb_epochs=10, image_size=320, early_stopping_patience=3,
                 n_valid_samples=2560,
                 batch_size=16, depth=4, channels=32, n_blocks=2, augment_images=True, debug_sample_size=None):
        self.model_name = 'resnet'
        self.weight_file_path = MODEL_BINARIES_PATH + self.model_name + '.h5'
        self.n_valid_samples = n_valid_samples
        self.nb_epochs = nb_epochs
        self.image_size = image_size
        self.augment_images = augment_images
        self.batch_size = batch_size
        self.depth = depth
        self.channels = channels
        self.n_blocks = n_blocks

        tb_callback = TensorBoard(log_dir=TB_LOGS_PATH, histogram_freq=0, write_graph=True,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        self.callbacks = [tb_callback, PlotLossesKeras()]
        self.callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience))
        self.callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                                patience=2,
                                                verbose=1,
                                                factor=0.1,
                                                min_lr=0.0001))
        self.callbacks.append(ModelCheckpoint(self.weight_file_path, monitor='val_loss', save_best_only=True))

        if debug_sample_size is not None:
            self.debug_sample_size = debug_sample_size
        self.load_data()
        if encoder_model == EncoderModelType.resnet:
            self.model = self.create_resnet_network(input_size=self.image_size,
                                                    channels=self.channels,
                                                    n_blocks=self.n_blocks,
                                                    depth=self.depth)
        elif encoder_model == EncoderModelType.vgg16:
            self.model = self.create_vgg16_network()
        self.model.compile(optimizer='adam',
                           loss=iou_bce_loss,
                           metrics=['accuracy', mean_iou])

    def create_downsample(self, channels, inputs):
        x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
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
        x = keras.layers.SpatialDropout2D(0.5)(x)
        return keras.layers.add([x, inputs])

    def create_vgg16_network(self, depth=4):
        # create the base pre-trained model
        base_model = VGG16(weights='imagenet', include_top=False)
        x = base_model.output

        # output
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        outputs = keras.layers.UpSampling2D(2 ** depth)(x)
        model = keras.Model(inputs=base_model.input, outputs=outputs)
        return model

    def create_resnet_network(self, input_size, channels, n_blocks=2, depth=4):
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
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0)(x)
        x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        outputs = keras.layers.UpSampling2D(2 ** depth)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_network(self):
        # create network and compiler
        model = self.create_network(input_size=self.image_size, channels=self.channels, n_blocks=self.n_blocks,
                                    depth=self.depth)
        model.compile(optimizer='adam',
                      loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])
        return model

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
            self.model.compile(optimizer='adam', loss=iou_bce_loss, metrics=['accuracy', mean_iou])
        return self

    def train(self):
        # create train and validation generators
        train_gen = generator(TRAIN_IMAGES_PATH, self.train_filenames, self.pneumonia_locations,
                              batch_size=self.batch_size,
                              image_size=self.image_size,
                              shuffle=True,
                              augment=self.augment_images, predict=False)
        valid_gen = generator(TRAIN_IMAGES_PATH, self.valid_filenames, self.pneumonia_locations,
                              batch_size=self.batch_size,
                              image_size=self.image_size,
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
                                 image_size=self.image_size,
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
            logging.info("Submission file is ready, good luck!")


if __name__ == "__main__":
    resnet = EncoderDecoderModel(nb_epochs=5)

    resnet.train()
    resnet.generate_submission()

# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_FOLDER_PATH = '/run/media/two-a-day/Elements/archive/data/pneumonia/all/'
TARGET_LABELS_DATA_PATH = DATA_FOLDER_PATH + 'stage_1_train_labels.csv'
DETAILED_CLASS_INFO_DATA_PATH = DATA_FOLDER_PATH + 'stage_1_detailed_class_info.csv'
TRAIN_IMAGES_PATH = DATA_FOLDER_PATH + 'stage_1_train_images/'
TEST_IMAGES_PATH = DATA_FOLDER_PATH + 'stage_1_test_images/'
SUBMISSIONS_FOLDER_PATH = 'submissions/'
MODEL_BINARIES_PATH = 'model_bins/'
LOGS_PATH = 'tmp/tb_logs/'

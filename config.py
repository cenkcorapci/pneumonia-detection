# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_FOLDER_PATH = '/run/media/two-a-day/Elements/archive/data/pneumonia/all/'
TARGET_LABELS_DATA_PATH = DATA_FOLDER_PATH + 'stage_1_train_labels.csv'
DETAILED_CLASS_INFO_DATA_PATH = DATA_FOLDER_PATH + 'stage_1_detailed_class_info.csv'
TRAIN_IMAGES_PATH = DATA_FOLDER_PATH + 'stage_1_train_images/'
DEMO_DATA_PATH = 'data/demo.csv'
LOGS_PATH = 'logs/logs'

PRETRAINED_EMBEDDINGS_PATH = 'bin/models/embedding.txt'

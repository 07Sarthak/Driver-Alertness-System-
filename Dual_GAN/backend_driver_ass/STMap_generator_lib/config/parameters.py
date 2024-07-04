import os
import time
from backend_driver_ass.STMap_generator_lib.tools.io_tools import mkdir_if_missing
#
# todo
#
from backend_driver_ass.directory import sample_location,OUTPUT_DIR
#DBS_DIR = r"C:\Users\sarth\Desktop\ML\New folder\collected_data"
sample_location=sample_location      #video and ground truth to be saved in SAMPLE_DATA folder
OUTPUT_DIR =OUTPUT_DIR
mkdir_if_missing(OUTPUT_DIR)
#
# database dir
#
#UBFC_rPPG_DATASET_2_DIR = os.path.join(DBS_DIR)
SAMPLE_LOCATION=sample_location                 #path to the the capture person gt and video
#PURE_DIR = os.path.join(DBS_DIR, 'PURE')
#VIPL_DIR = os.path.join(DBS_DIR, 'VIPL_HR_V1')
#MAHNOB_HCI_DIR = os.path.join(DBS_DIR, 'MAHNOB_HCI', 'Sessions')
#
# config
#
CFG = {
    'MODEL_SAVE_DIR': os.path.join(OUTPUT_DIR, 'model_save') + '_' + time.strftime('%Y-%m-%d-%H-%M-%S'),
    'SEED': 1024,
    'DEVICE_ID': '4',
    'PRE_TRAIN_CHOICE': '',
    'PRE_TRAIN_PATH': '',
    'train_txt_path': os.path.join(OUTPUT_DIR, 'train.txt'),
    'valid_txt_path': os.path.join(OUTPUT_DIR, 'valid.txt'),
    'EPOCH_NUM': 40,
    'LOG_PERIOD': 50,
    'SAVE_PERIOD': 1,
    'EVAL_PERIOD': 1,
    'lr': 0.0001,
    'TRAIN_BATCH_SIZE': 32,
    'VALID_BATCH_SIZE': 32,
}

if __name__ == '__main__':
    #assert os.path.isdir(DBS_DIR)
    assert os.path.isdir(SAMPLE_LOCATION)
    #assert os.path.isdir(PURE_DIR)
    #assert os.path.isdir(MAHNOB_HCI_DIR)
    #assert os.path.isdir(VIPL_DIR)
    #assert os.path.isdir(OUTPUT_DIR)

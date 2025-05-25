import getpass
import os
import sys

_BASE_DIR = '../dataset'

DATA_FOLDER = os.path.join(_BASE_DIR, 'raw_data')
LLAMA_PATH = '/your_LLAMA_path/'
DATA_PATH = '/your_dataset_path/'

GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'quantization')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
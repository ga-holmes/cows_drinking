# 'make_data.py': Sample file using dataset functions
#     - Given the information from 'read_csv' or a json file ('all_data.json') 
#       & a directory containing the corresponding video files, 
#       generates a dataset in numpy format & outputs it to a '.npy' file.
#     - Using default class configuration

from functions.dataset_functions import *
from functions.read_csv import read_csv

videos_path = 'videos'

# Creating the dataset using CowsWater class:

# When creating the class, you can either use the labels in dictionary format as created here directly,
# or you can use a file in JSON format
csv_data = read_csv('data/data.csv', 'data/recording_codes.csv')

# example + default values that can be changed:
# To use a file, set 'load_from_file' to True, & include a filename in the 'label_path' parameter
cowsdrink = CowsWater(videos_path, data=csv_data, grayscale=False, shuffle=True, img_size=256, load_from_file=False)

# format expected for the labels (This should not be changed, the class will only work with these labels exactly).
# to create a dataset for a different format, write a new class based on this one.
cowsdrink.LABELS = {"EMPTY": 0, "COW": 1}

# ex. to create a dataset with colour (MUCH larger file created by this)
cowsdrink.SAVE_NAME = 'dataset.npy'

cowsdrink.make_training_data(step_param=180, valid_step_param=24)
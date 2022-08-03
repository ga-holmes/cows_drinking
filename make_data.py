from dataset_functions import *

videos_path = 'videos_2'

# Creating the dataset using CowsWater class:

# example + default values that can be changed
cowsdrink = CowsWater(videos_path, 'all_data.json', grayscale=False, shuffle=False, img_size=256)

# format expected for the labels (in this case, a binary classification)
cowsdrink.LABELS = {"EMPTY": 0, "COW": 1}

# ex. to create a dataset with colour (MUCH larger file created by this)
cowsdrink.SAVE_NAME = 'training_110-111_colour_256.npy'

cowsdrink.make_training_data(step_param=96, valid_step_param=6)
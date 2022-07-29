from dataset_functions import *

videos_path = 'videos'

# Creating the dataset using CowsWater class:

# example + default values that can be changed
cowsdrink = CowsWater(videos_path, 'all_data.j-son', grayscale=True, img_size=256)

# format expected for the labels (in this case, a binary classification)
cowsdrink.LABELS = {"EMPTY": 0, "COW": 1}

# ex. to create a dataset with colour (MUCH larger file created by this)
cowsdrink.GRAYSCALE = False
cowsdrink.SAVE_NAME = 'training_all_colour_256.npy'

cowsdrink.make_training_data(step_param=180, valid_step_param=24)
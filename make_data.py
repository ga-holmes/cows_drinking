from dataset_functions import *

cowsdrink = CowsWater()

cowsdrink.GRAYSCALE = False
cowsdrink.SAVE_NAME = 'D:/training_all_colour.npy'

cowsdrink.make_training_data()
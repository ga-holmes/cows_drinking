import json
import os
from statistics import mode
import cv2
import numpy as np
from tqdm import tqdm

import torch

import time
import datetime as dt

from torchvision import transforms

import imgaug as ia
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from zmq import device

# dataset functions:


# define a class for the dataset
class CowsWater():
    # Specify the image size (width & height)
    IMG_SIZE = 512
    # specify whether to convert to grayscale
    GRAYSCALE = True
    SAVE_NAME = 'training_all.npy'

    # directory locations
    VIDEOS = "videos"
    LABEL_PATH = "all_data.json"

    # define the labels
    LABELS = {"EMPTY": 0, "COW": 1}

    # list we will populate with images
    training_data = []

    # we want to keep track of the counts to ensure the dataset is balanced
    empty_count = 0
    cow_count = 0

    def __init__(self, input_path, label_path, grayscale=True, img_size=512):
        
        self.VIDEOS = input_path
        self.LABEL_PATH = label_path
        self.GRAYSCALE = grayscale
        self.IMG_SIZE = img_size

    # creates a training data file in the format (list) [(np.array - the image), (a one-hot vector containing the label for the image)]
    # step is how often to save an image from the video (default: every 180 frames)
    # valid_step is how often to save the image when the image is 'positive' (ie. a cow is there) (default: every 24 frames)
    # these are used to balance the dataset when there are more 'negative' frames than 'positive' frames
    def make_training_data(self, step_param=180, valid_step_param=24):

        print(f'Grayscale: {self.GRAYSCALE}, Save Name: {self.SAVE_NAME}, Img Size: {self.IMG_SIZE}')

        with open('all_data.json') as f:
            data = json.load(f)

        # so that these can be changed throughout the program
        step = step_param
        valid_step = valid_step_param

        # to save the original step
        set_step = step

        time_zero = dt.datetime.strptime('00:00:00', '%H:%M:%S')
        # iterate over all the images in the directory
        # NOTE: tqdm is just a progress bar
        # NOTE: Folder format expected: VIDEOS/DD-MM-YYYY/pen(s)/vido_file.MP4
        for date in os.listdir(self.VIDEOS):
            
            dir_date = date.replace('-', '/')
            dates = []
            dates = [dp for dp in data if dp['DATE'] == dir_date]

            print('\n\n'+dir_date)

            # all pens at date
            for pen in os.listdir(os.path.join(self.VIDEOS, date)):
                
                pens = []
                pens = [dp for dp in dates if ( dp['PEN'] in pen.split('-'))]

                # NOTE: don't include videos with multiple pens (for now)
                if '-' in pen:
                    continue

                print('\t'+pen)
                
                # all videos at pen for that date
                for f in os.listdir(os.path.join(self.VIDEOS, date, pen)):
                    files = []
                    files = [dp for dp in pens if dp['VIDEO']['FILE_NAME'] in f]

                    path = os.path.join(self.VIDEOS, date, pen, f)

                    # time regions to ignore (bad data)
                    ignore = []
                    for t in files[0]['VIDEO']['IGNORE']:
                        ignore.append([
                            dt.datetime.strptime(t[0], '%H:%M:%S').time(),
                            dt.datetime.strptime(t[1], '%H:%M:%S').time()
                        ])

                    # load the video from the path
                    cap = cv2.VideoCapture(path)

                    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # ret is a true or false value, false if no frame is returned
                    for i in tqdm(range(totalframecount)):
                        
                        # with steps of 1, just read the frame
                        if step < 2:    
                            ret, frame = cap.read()
                            
                            # break if there's no frames left        
                            if not ret:
                                break

                        # get the time in the video & convert to isoformat

                        # every {step} frames
                        if i % step == 0:
                            
                            curr_time =  dt.datetime.strptime(f'{int(((i/fps)/60)/60):02d}:{int(((i/fps)/60)%60):02d}:{int((i / fps)%60):02d}', '%H:%M:%S')
                            clock_time = (curr_time - time_zero + dt.datetime.strptime((files[0])['VIDEO']['START'], '%H:%M')).time()
                            curr_time = curr_time.time()

                            # check if we should ignore this frame (marked as bad data)
                            do_ignore = False
                            
                            for t in ignore:
                                if curr_time > t[0] and curr_time < t[1]:
                                    do_ignore = True

                            # ignore this frame if set to true
                            if not do_ignore:

                                # get the date from the json
                                dir_date = date.replace('-', '/')
                                
                                # for steps greater than 1, jump forward
                                if step > 1:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                                    ret, frame = cap.read()

                                    # break if there's no frames left        
                                    if not ret:
                                        break
                                
                                # resize the images so they can be input into the network
                                img = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))

                                # convert to grayscale OR switch to RGB colour (instead of BGR)
                                if self.GRAYSCALE:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                else:
                                    img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
                                
                                # list of datapoints for the frame
                                dps = [dp for dp in files if clock_time > (dt.datetime.strptime(dp['START'], '%H:%M:%S')).time() and clock_time < (dt.datetime.strptime(dp['END'], '%H:%M:%S')).time()]
                                
                                # initially set to empty
                                label = "EMPTY"
                                if len(dps) > 0: # if there is a data point here, the frame is positive
                                    label = "COW"
                                    self.cow_count += 1
                                    # lower the step to get more images:
                                    step = valid_step
                                    # cv2.imshow('Window', img)
                                    # cv2.waitKey(0)
                                else:
                                    self.empty_count += 1
                                    # reset the step
                                    step = set_step
                                
                                #  np function to convert a dict/array like we have into a 1 hot vector
                                #  (#) is the length of the vector (num classes), [#] is the activated index of the vector
                                vec = np.eye(2)[self.LABELS[label]]

                                # appending the data to the training_data array
                                # essentially putting it into the same format as the MNIST FCN
                                # img is converted to a numpy array, the second parameter is the class label (in this case we make a 1-hot vector)
                                self.training_data.append([np.array(img), vec])
                
                cap.release()   

        # shuffle the dataset
        np.random.shuffle(self.training_data)
        
        # save the training data
        np.save(self.SAVE_NAME, self.training_data)

        # check the balance
        print("Cows:", self.cow_count)
        print("Empty:", self.empty_count)


# loads image+label data from 'filename', resizes it to 'im_size', applies 'preprocessing;, 
# & splits into training & test sets, were the test set is 'validation_percent' the size of the training set, also returns image information dictionary
# NOTE: Data is expected in the following format:
# A .npy file containing a list of numpy arrays representing images, along with corresponding labels in one-hot vector format
# Generally, follow the format that the CowsWater class saves the data in
def load_split_data(filename, preprocessing=iaa.Sequential, validation_percent=0.1):
    # load from file
    training_data = np.load(filename, allow_pickle=True)

    print('done loading')

    # convert to tensor for each image & adjust image scaling
    X = []
    for i in tqdm(training_data):
        # convert colours to rgb if they are in grayscale (pre-trained models only accept 3-channel images)
        im = cv2.cvtColor(i[0], cv2.COLOR_GRAY2RGB) if len(i[0].shape) < 3 else i[0]
        # apply the image transforms (one image at a time to save memory)
        im = preprocessing(image=im)

        X.append(im)

    # convert to tensor for each image & flatten
    # X = torch.Tensor([i[0] for i in training_data])

    imsize = X[0].shape[1]
    X = np.array(X)

    num_chan = X[0].shape[2]
    X = torch.from_numpy(X).float().permute(0,3,1,2)


    # scale the values in the images from 0 to 255 to 0 to 1 only if they are not already at that scale
    if X.max() > 1:
        X = X/255.0

    print('done preprocessing')

    # seperate the Xs & the ys from the training_data list

    # get all the labels
    y = torch.Tensor([i[1] for i in training_data])

    # how many images?
    val_size = int(len(X) * validation_percent)

    # simply splice the arrays
    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    im_info = {
        'classes': len(y[0]),
        'channels': num_chan,
        'size': imsize
    }

    return train_X, train_y, test_X, test_y, im_info


# Remove characters that aren't allowed in a filename
def fix_filename(filename: str):
    bad_chars = ['*', ':', '\"', '<', '>', '|', '?']
    for c in bad_chars:
        filename = filename.replace(c, '')
    return filename

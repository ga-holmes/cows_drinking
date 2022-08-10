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

# model functions:

class Trainer():

    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()

    # if there is a memory error, lower the batch size (you want a batch size of at least 8 however)
    BATCH_SIZE = 100
    EPOCHS = 5
    device_to_use = 'cpu'

    log_dir = ''

    def __init__(self, model, im_size, modelname, device_to_use='cpu', optimizer=optim.SGD, im_chan=3, scheduler=None):
        self.model = model
        self.optimizer = optimizer(self.model.fc.parameters(), lr=self.learning_rate)
        self.device_to_use = device_to_use
        self.im_size = im_size
        self.im_chan = im_chan
        self.modelname = modelname
        self.scheduler = scheduler

    # function to take data & put it through the model, if train=True, update the parameters as well
    def fwd_pass(self, X, y, train=False):
        
        # only zero the gradients if we are training
        if train:
            self.model.zero_grad()

        outputs = self.model(X)

        # find out accuracy
        # for all the results, count how many predictions match the ground truth (true) vs dont (false)
        matches = []
        bad_results = []
        bad_labels = []
        n = 0
        for i, j in zip(outputs, y):
            matches.append(torch.argmax(i) == torch.argmax(j))
            if not torch.argmax(i) == torch.argmax(j):
                bad_results.append(X[n])
                bad_labels.append(torch.argmax(i))

            n += 1

        # how many true matches over the length of matches
        accuracy = matches.count(True)/len(matches)

        loss = self.loss_function(outputs, y)

        # only update if training
        if train:
            loss.backward()
            self.optimizer.step()

        return accuracy, loss, bad_results, bad_labels


    # runs the fwd_pass function in a loop by 'epochs', with the dataset seperated by 'batch_size'
    # when validate is set to true, the model will do a forward pass on the test set
    # the loss & accuracy for val & train will also be written to a log file every 'val_steps' iterations over the batch
    # if validate is true, test_X & test_y must be defined
    def train(self, train_X, train_y, validate=False, val_steps=50, test_X=[], test_y=[]):

        with open(os.path.join(self.log_dir, f"{self.modelname}.log"), "a") as f:

            for epoch in range(self.EPOCHS):
                for i in tqdm(range(0, len(train_X), self.BATCH_SIZE)):

                    # put the batches through the network (sliced based on batch size)
                    batch_X = train_X[i:i+self.BATCH_SIZE].view(-1, self.im_chan, self.im_size, self.im_size).to(self.device_to_use)
                    batch_y = train_y[i:i+self.BATCH_SIZE].to(self.device_to_use)

                    # do a forward pass with training on to get accuracy & loss
                    acc, loss, bad_results, bad_labels = self.fwd_pass(batch_X, batch_y, train=True)

                    # validation during training, saved to the model.log file
                    if validate and len(test_X) > 0 and len(test_y) > 0 and i % val_steps == 0:
                        
                        val_acc, val_loss, bad_results, bad_labels = self.test(test_X, test_y, size=100)
                        # write the model name, the time, the accuracy, & the loss to the model.log file
                        f.write(f"{self.modelname},{round(time.time(), 3)}, {round(float(acc), 2)}, {round(float(loss), 4)}, {round(float(val_acc), 2)}, {round(float(val_loss), 4)}, {epoch}\n")
                        
                # step the learning rate if not none
                if self.scheduler is not None:
                    self.scheduler.step()


    # tests some data (default 32 pieces of data)
    def test(self, test_X, test_y, size=32):

        random_start = np.random.randint(len(test_X)-size)

        # get subsection of the dataset for testing
        X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]

        with torch.no_grad():
            val_acc, val_loss, bad_results, bad_labels = self.fwd_pass(X.view(-1, self.im_chan, self.im_size, self.im_size).to(self.device_to_use),y.to(self.device_to_use), train=False)

        return val_acc, val_loss, bad_results, bad_labels
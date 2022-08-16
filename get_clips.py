
# Python script that will automatically create clips of sections 
# in a given video where there is an animal in the drinking area.
# The script takes a file name as input (plus other options) & 
# saves *N clips from the video in a folder (output directory can also be specified)
# 
# Input:
#   -f: filename/path to file (ex. C:/videos/drinking.mp4
#   -o: output directory (ex. C:/videos/clips
#   -wf: file path for trained model weights 
# 
# Options: 
#   -s: step 
#       (how many frames to skip when no cow present)
#   -sv: valid step 
#       (how many frames to skip when a cow is present)
#   -imsize: what dimension to resize the image to 
#       (should be the same as the size used for training)
#   -e: 'Allowed Error' - how many ‘no cow’ predictions to allow before 
#       stopping saving to the current clip 
#       (Allows the model to make some incorrect predictions)
#   -ml: minimum length in frames of a clip that can be saved 
#       (avoids saving clips that are only a few seconds long)



import cv2
import matplotlib.pyplot as plt

from functions.trainer import *
from functions.dataset_functions import get_video_time

import torch
from torchvision.models import resnet50, resnet18, resnext50_32x4d

import time
import sys, getopt

FILENAME = None
output_dir = ''

# specify a file path for the model weights here or from cmd args
WEIGHT_FILE = 'weights/resnet18_colour_local_scheduled_10e128bsMSE_Adam_weights.pth'

# set the default values here or in the command line when running the program
step = 48
val_step=4
im_size = 256
allowed_error = 48
min_length = 96

# simple function for cmd line args
def num_or_zero(s:str, default):
    if s.isdigit():
        return int(s)
    else:
        return default

# get command line arguments
for i, arg in enumerate(sys.argv):
    if arg == '-f' and i < len(sys.argv)-1:
        FILENAME = sys.argv[i+1]
    elif arg == '-o' and i < len(sys.argv)-1:
        output_dir = sys.argv[i+1]
    elif arg == '-s' and i < len(sys.argv)-1:
        step = num_or_zero(sys.argv[i+1], step)
    elif arg == '-sv' and i < len(sys.argv)-1:
        val_step = num_or_zero(sys.argv[i+1], val_step)
    elif arg == '-imsize' and i < len(sys.argv)-1:
        im_size = num_or_zero(sys.argv[i+1], im_size)
    elif arg == '-e' and i < len(sys.argv)-1:
        allowed_error = num_or_zero(sys.argv[i+1], allowed_error)
    elif arg == '-ml' and i < len(sys.argv)-1:
        min_length = num_or_zero(sys.argv[i+1], min_length)
    elif arg == '-wf' and i < len(sys.argv)-1:
        WEIGHT_FILE = sys.argv[i+1]


if FILENAME is None:
    print('please include \'-f [FILENAME]\'')
    exit()

# verify args
print(f'{FILENAME}: step={step}, val_step={val_step}, im_size={im_size}, allowed_error={allowed_error}, min_length={min_length}, weights={WEIGHT_FILE}')

model = resnet18(pretrained=False)
num_classes = 2
LOG_NAME = 'clip_model'


# dynamically define whether to run on gpu or cpu
device_to_use = torch.device("cpu")

if torch.cuda.is_available():
    device_to_use = torch.device("cuda:0")
    print("Running on GPU")
else:
    print("Running on CPU")


def get_args(argv):
    
    print(len(argv))


# parses the given video &  uses 'model' to create a list of lists of sections of the video 
# where the BINARY model predicted a true value (1).
# Moves every 'step' frames for false predictions, & 'val_step' frames for true predictions
# Adding to a clip is only stopped after 'allowed_error' frames of false predictions
# this allows the model to be wrong sometimes (no model is 100% accurate)
def clip_vid(video_path, model, step=24, val_step=2, im_size=256, allowed_error=8):
    
    # save the step value
    use_step = step

    # load the video from the path
    cap = cv2.VideoCapture(video_path)

    totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    clips = []
    times = []

    prev_frame = False
    false_count = 0
    clip_i = -1

    # ret is a true or false value, false if no frame is returned
    for i in tqdm(range(totalframecount)):
        
        # every {step} frames
        if i % use_step == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)

            ret, frame = cap.read()

            # break if there's no frames left        
            if not ret:
                break

            frame = cv2.cvtColor(cv2.resize(frame, (im_size, im_size)), cv2.COLOR_BGR2RGB)

            # cv2.imshow('test', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            image = torch.from_numpy(frame).float().permute(2,0,1)

            if image.max() > 1:
                image = image/255.0

            result = model(image.to(device_to_use).unsqueeze(0))
            result = torch.argmax(result)

            if result == 1 or false_count < allowed_error:

                if result != 1: false_count += 1

                # if this is a new clip
                if not prev_frame:
                    clips.append([])
                    times.append([get_video_time(i,fps)])

                    clip_i += 1

                    # update values
                    prev_frame = True
                    use_step = val_step
                    false_count = 0

                clips[clip_i].append(frame)
                  
            else:

                # only move on to new clip if its been more than 'allowed_error' frames false
                false_count += 1
                if false_count > allowed_error:
                    use_step = step

                    if prev_frame:
                        times[clip_i].append(get_video_time(i,fps))

                    prev_frame = False

    
    cap.release()   
                
    return clips, times, fps / val_step


# Given a list if lists containing images in the form of numpy arrays,
# saves all lists of images as videos above 'min_clip_length' frames in length
# at 'fps' frames per second
def save_clips(clips, fps, min_clip_length=24):

    # only create a clip if theere are frames
    if len(clips) > 0:

        for i, clip in enumerate(clips):
            
            # ignore clips that have less frames than the minimum amount
            if len(clip) < min_clip_length:
                continue

            # writing 'clips' to video
            out = cv2.VideoWriter(f'{output_dir}/clip_{i}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (im_size, im_size))

            # writing to the video
            for f in tqdm(clip):
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

            out.release()


# main operations
if __name__ == "__main__":

    # loading the model

    # Get the number of inputs to the last fully connected layer in the model
    num_ftrs = model.fc.in_features
    # create a new final fully connected layer that we can train to replace the other fully connected layer
    model.fc = nn.Linear(num_ftrs, num_classes)


    # add the model to the device
    model = model.to(device_to_use)

    model.load_state_dict(torch.load(WEIGHT_FILE, map_location=device_to_use))

    cow_model = Trainer(model, im_size, LOG_NAME, device_to_use=device_to_use)

    cow_model.loss_function = nn.MSELoss()
    cow_model.optimizer = optim.Adam(model.fc.parameters(), lr=cow_model.learning_rate)

    print(f"Created {LOG_NAME}")


    model.eval()

    # Getting the clips
    
    clips, times, fps = clip_vid(FILENAME, model, step=step, val_step=val_step, im_size=im_size, allowed_error=allowed_error)

    save_clips(clips, fps, min_clip_length=min_length)

    print(len(clips), 'clips found')

    # min_time = dt.datetime.strptime(f'{int((min_length / fps)):02d}', '%S').time()

    # show the intervals of time
    for i, t in enumerate(times):

        # only show intervals that were actually saved
        if len(clips[i]) >= min_length:
            print(f'clip {i}: {t[0]} - {t[1]}')

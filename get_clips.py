import cv2
import matplotlib.pyplot as plt

from trainer import *
from dataset_functions import get_video_time

import torch
from torchvision.models import resnet50, resnet18, resnext50_32x4d

import time

WEIGHT_FILE = 'weights/resnet18_colour_local_scheduled_10e128bsMSE_Adam_weights.pth'
model = resnet18(pretrained=False)
LOG_NAME = 'clip_model'
im_size = 256

num_classes = 2

# dynamically define whether to run on gpu or cpu
device_to_use = torch.device("cpu")

if torch.cuda.is_available():
    device_to_use = torch.device("cuda:0")
    print("Running on GPU")
else:
    print("Running on CPU")


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


# parses the given video & 
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

def save_clips(clips, fps, min_clip_length=24):

    # only create a clip if theere are frames
    if len(clips) > 0:

        for i, clip in enumerate(clips):
            
            # ignore clips that have less frames than the minimum amount
            if len(clip) < min_clip_length:
                continue

            # writing 'clips' to video
            out = cv2.VideoWriter(f'clips/clip_{i}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (im_size, im_size))

            # writing to the video
            for f in tqdm(clip):
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

            out.release()

clips, times, fps = clip_vid('test_clips/test2.mp4', model, step=48, val_step=4, im_size=256, allowed_error=48)

min_length = 96
save_clips(clips, fps, min_clip_length=min_length)

print(len(clips), 'clips found')

# min_time = dt.datetime.strptime(f'{int((min_length / fps)):02d}', '%S').time()

# show the intervals of time
for i, t in enumerate(times):

    # only show intervals that were actually saved
    if len(clips[i]) >= min_length:
        print(f'clip {i}: {t[0]} - {t[1]}')


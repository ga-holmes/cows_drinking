Date: July 2022
Author: Garrett Holmes
### Description: Summer 2022 UoGuelph URA project - automatic detection of cattle in drinking areas using CNNs
---
## Requirements:

- Python version 3.7+
- Uses PyTorch for CNNs & OpenCV for initially processing the data into numpy format. ImgAug for augmentations.
- Install all requirements using directions below

### Installation:
- With Conda: `conda create --name <env> --file requirements.txt`
- With Pip (must have python/pip installed already): `pip install -r requirements.txt`
---
## Usage:
### Files:
- 'read_csv.py':
    - Reads the data from the given csv files expecting the format presented by the files in /data/ (One file for observations (data points), one for video file information)
    - Relates the data points to their corresponding videos & outputs a JSON file containing the compiled information (all_data.json)
- 'make_data.py':
    - Given the information in 'all_data.json' & a directory containing the corresponding video files, generates a dataset in numpy format & outputs it to a '.npy' file.
    - Using default class configuration
- 'dataset_functions.py':
    - Specifics of the dataset generation (Colour vs Grayscale, Image Size, Input/Outputs paths, etc.) can all be altered directly in the 'CowsWater' class in  the 'dataset_functions.py' file
    - TODO: Make the class information more readily customizable
- 'trainer.py':
    - Contains functions for the 'Trainer' class, which is used to train & test the chosen CNN
- 'resnet.ipynb':
    - Loads one of 2 ResNet models from PyTorch - both ResNet50 & ResNet18 work & can be specified in the first cell
    - `DOWNLOAD_WEIGHTS` specifies whether or not to simply load pretrained weights (`True`) or load weights from a file (`False`)
        - The 'resnet18_weights_alldata.pth' file is provided & will load weights trained on a dataset made from only videos with 1 visible pen, and will only work on ResNet18
    - Loads the dataset from the given '.npy' file
    - Loads the model & creates a Trainer class
    - Can either run the training cell to train the model (not recommended if not on GPU) or just the cell that tests the network on the test dataset
        - You can test without training and get good results if you load the .pth file (`DOWNLOAD_WEIGHTS=False`)
    - Final cell creates a window that will display all the incorrect predictions from the last test run

### Using the 'Trainer' class (See 'resnet.py' as an example)
- Initialize:
    - `<name> = Trainer( params )`
        - Required parameters: model, im_size, modelname
            - model: a PyTorch model (ex. resnet18())
            - im_size: image dimensions in pixels
            - modelname: a string representing the name of the model for saving weights & log file
        - Optional parameters: device, optimizer, im_chan
            - device: default 'cpu', set to run with a GPU
            - optimizer: default optim.SGD, set to use a different optimizer (idk if it will work)
            - im_chan: will always be 3 for PyTorch pretrained models
    - `<name>.BATCH_SIZE` | `<name>.EPOCHS` | `<name>.learning_rate`: set these values after definition, defaults are 100, 5, 0.001 respectively
    - `<name>.loss_function`: set to use a specific loss function from torch.nn. Default = 'nn.CrossEntropyLoss()'
- Usage:
    - `<name>.train(train_X, train_y)`: Trains the network & saves loss & accuracy information to a '<modelname>.log' file
        - train_X & train_y: Input images & corresponding labels respectively (required)
        - validate=False: whether or not to collect accuracy & loss from a validation set during training & save to a log file (requires test_X & test_y to be not empty)
        - val_steps=50: How often to validate & save to the log file
        - test_X, test_y = []: Images & labels in the same format as X & y for validation (make sure the data is independent of the training set)
        - Returns: nothing
    - `<name>.test(test_X, test_y)`: Runs the input data through the network without calculating gradients or updating weights
        - test_X, test_y: Input images & corresponding labels respectively (required)
        - size=32: How big of a random sample from the given test sets to take ('size' of the total set of inputs will be randomly selected)
        - Returns: 
            - val_acc: Accuracy on the set
            - val_loss: Loss on the set
            - bad_results: A list of images that had incorrect predictions
            - bad_labels: A list of the incorrect prediction labels corresponding to the images
---
### Expected 'videos' directory format for CowsWater() class

- /videos / DD-MM-YYYY / PEN# / FILENAME.MP4

- [There can be multiple date/pen folders inside the videos folder, and multiple videos inside the pen folders]
- ex:
    >videos/\
    ├─ DD-MM-YYYY/\
    │  ├─ PEN# (ex. 101)/\
    │  │  ├─ FILENAME.MP4\
    │  ├─ 118/\
    │  │  ├─ FILENAME.MP4\
    │  │  ├─ FILENAME.MP4\
    ├─ 11-02-2022/\
    │  ├─ 120-121/\
    │  │  ├─ FILENAME.MP4

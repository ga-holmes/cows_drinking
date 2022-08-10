Date: July 2022
Author: Garrett Holmes
### Description: Summer 2022 UoGuelph URA project - automatic detection of cattle in drinking areas using CNNs
---
## Requirements:

- Python version 3.7+
- Uses PyTorch for CNNs & OpenCV for initially processing the data into numpy format. ImgAug for augmentations.
- Install all requirements using directions below

### Installation:(must have python/pip 3.7+ installed already)
- With Conda: `conda create --name <env> --file requirements.txt`
- With Pip: `pip install -r requirements.txt`

- Must Install Pytorch Seperately: 
    - go to https://pytorch.org/ & navigate to install table
    - Choose the appropriate PyTorch build for your system (whether you want CUDA, etc.)
    - ex. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

---
## Usage:
### Files:
- 'read_csv.py':
    - Reads the data from the given csv files expecting the format presented by the files in /data/ (One file for observations (data points), one for video file information)
    - Relates the data points to their corresponding videos & outputs a JSON file containing the compiled information (all_data.json)
- 'make_data.py':
    - Given the information in 'all_data.json' & a directory containing the corresponding video files, generates a dataset in numpy format & outputs it to a '.npy' file.
    - Using default class configuration
- 'dataset_functions.py': Contains functions with regards to data management & preprocessing.
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
- 'graph.py':
    - Creates a visualization of the loss & accuracy over the course of training
        - Must provide a filename the bottom of the file
        - Make sure the `model_name` variable is the same as the name in the log file

### Using the 'Trainer' Class: 
- See readme in functions folder

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

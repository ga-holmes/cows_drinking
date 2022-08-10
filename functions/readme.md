
## Using the 'Trainer' Class 

- (See 'resnet.py' as an example)

### Initialize:
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

---

### Usage:
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
# 'graph.py':
#     - Creates a visualization of the loss & accuracy over the course of training
#         - Must provide a filename the bottom of the file
#         - Make sure the `model_name` variable is the same as the name in the log file
#     - Arguments: Specify the file name / path to use in the command line when running the program
#         - graph.py -f [FILENAME/PATH]

import sys

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# .log file name/path
fname = None

# get command line arguments
# Usage: graph.py -f [FILENAME/PATH]
for i, arg in enumerate(sys.argv):
    if arg == '-f' and i < len(sys.argv)-1:
        fname = sys.argv[i+1]

if fname is None:
    print('please include \'-f [FILENAME]\'')
    exit()

# graphs the accuracy and loss of the model
def create_acc_loss_graph(file_name):
    # open the file, seperate by newline
    contents = open(file_name, 'r').read().split('\n')

    times = []
    accuracies = []
    losses = []

    val_accuracies = []
    val_losses = []

    # iterate through all the lines in the file
    for i, c in enumerate(contents):
        # make sure the line is of the right model name
        if len(c.split(',')) > 6:
            # get the values in each line (c) seperated by comma
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(',')

            # add them to the list
            times.append(float(i))
            accuracies.append(float(acc))
            losses.append(float(loss))
            
            val_accuracies.append(float(val_acc))
            val_losses.append(float(val_loss))

    # graph the lists
    fig = plt.figure() # for defining multiple graphs

    # 2x1 grid
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

    ax1.plot(times, accuracies, label='acc')
    ax1.plot(times, val_accuracies, label='val_acc')
    ax1.legend(loc=2)

    ax2.plot(times, losses, label='loss')
    ax2.plot(times, val_losses, label='val_loss')
    ax2.legend(loc=2)

    plt.show()

    
create_acc_loss_graph(file_name=fname)

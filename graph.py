import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

model_name = 'resnet18_alldata_finetuned'

# graphs the accuracy and loss of the model
def create_acc_loss_graph(model_name, file_name):
    # open the file, seperate by newline
    contents = open(file_name, 'r').read().split('\n')

    times = []
    accuracies = []
    losses = []

    val_accuracies = []
    val_losses = []

    # iterate through all the lines in the file
    for c in contents:
        # make sure the line is of the right model name
        if model_name in c:
            # get the values in each line (c) seperated by comma
            name, timestamp, acc, loss, val_acc, val_loss = c.split(',')

            # add them to the list
            times.append(float(timestamp))
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

    
create_acc_loss_graph(model_name, file_name='resnet18_alldata_finetuned.log')

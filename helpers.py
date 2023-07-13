from scipy.io import loadmat
import os
import numpy as np
from mnist import MNIST

def load_953_dataset(data_directory):
    """
    loads dataset from 9.53: 3000 training and testing examples. label 10 is really a 0
    note to self: to access the ith training example: do this training_datapoints[:,[i]].

    we should convert 0s to 10s later to make the design better.
    """
    training_data = loadmat(os.path.join(data_directory, 'mnist_training.mat'))
    training_datapoints, training_labels = training_data['X_train'], training_data['y_train']
    testing_data = loadmat(os.path.join(data_directory, 'mnist_test.mat'))
    testing_datapoints, testing_labels = testing_data['X_test'], testing_data['y_test']

    # converting labels to be a list of ints, to fit the desired format:
    testing_labels = np.reshape(testing_labels, (len(testing_labels)))
    training_labels = np.reshape(training_labels, (len(training_labels)))

    # making each column the datapoint, not the row:
    training_datapoints = training_datapoints.T
    testing_datapoints = testing_datapoints.T

    return training_datapoints, training_labels, testing_datapoints, testing_labels
    
def load_full_dataset(data_directory):
    """
    loads the complete dataset: 60000 training and 10000 testing examples. 0s are labelled as 0 here.
    """
    mndata = MNIST(data_directory)
    training_datapoints, training_labels = mndata.load_training()
    testing_datapoints, testing_labels = mndata.load_testing()

    # centering data to the range of [0,1], rather than [0, 255]:
    training_datapoints = np.array(training_datapoints)/255
    testing_datapoints = np.array(testing_datapoints)/255

    # making each column the datapoint, not the row:
    training_datapoints = training_datapoints.T
    testing_datapoints = testing_datapoints.T

    return training_datapoints, training_labels, testing_datapoints, testing_labels
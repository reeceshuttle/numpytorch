import numpy as np
import matplotlib.pyplot as plt
import time
import os
import numpytorch as npt
from types import SimpleNamespace


        
def get_metrics(current_network: npt.NeuralNetwork, points, labels):
    """will return classification error and total loss
    for both training and testing sets.
    labels should be a list of ints. preprocessing should occur before input here?
    """
    total_incorrect = 0
    for i in range(len(labels)):
    # for point, label_ in zip(points, labels):
        label = labels[i]%10 # since 10 is 0's label
        point_in = points[:,[i]]
        prediction = current_network.forward(point_in)
        predicted_class = np.argmax(prediction)
        if predicted_class != label:
            total_incorrect += 1
    return total_incorrect/len(points[0])

def train(network: npt.NeuralNetwork, dataloader: npt.MNISTDataLoader, 
          optimizer: npt.Optimizer, stochastic_steps: int, 
          get_progress: bool = True, progress_step: int = 10000, 
          get_print_statements: bool = True, print_step: int = 100000):
    """
    progress step means that if get_progress is True, you do the training and testing loss and performance validation at
    every progress_step steps.
    """
    training_data, training_labels = dataloader.training_set()
    corresponding_step = []
    if get_progress:
        testing_data, testing_labels = dataloader.testing_set()
        testing_errors = []
        training_errors = []
    else:
        testing_errors = None
        training_errors = None
        
    training_start = time.time()
    for step_num in range(stochastic_steps):
        if step_num%print_step==0 and step_num!=0 and get_print_statements:
            print(f'reached step {step_num} in {time.time()-training_start} sec', end='\r')

        if step_num%progress_step==0 and get_progress:
            testing_error_proportion = get_metrics(network, testing_data, testing_labels)
            testing_errors.append(testing_error_proportion)
            training_error_proportion = get_metrics(network, training_data, training_labels)
            training_errors.append(training_error_proportion)
            corresponding_step.append(step_num)

        i = np.random.randint(0, 3000-1)
        point, label = training_data[:,[i]], training_labels[i]
        estimate = network.forward(point)
        dLdZ = npt.SquaredLoss.gradient(estimate, npt.convert_label_to_vect(label))
        network.backward(dLdZ, lr=0.001)
    if get_print_statements:
        print('                                                   ', end='\r')
        print(f'training time:{time.time()-training_start} sec')
    data = {'testing_errors':testing_errors, 'training_errors':training_errors,
            'corresponding_step':corresponding_step}
    return SimpleNamespace(**data)



if __name__ == "__main__":
    np.random.seed(0)
    nn_sequential = [npt.Linear(784, 20, last_linear=True), npt.ReLU(), 
                     npt.Linear(20, 10), npt.Softmax()]

    
    network = npt.NeuralNetwork(nn_sequential)
    optimizer = npt.VanillaOptimizer(network)
    # network.initialize_optimizer(npt.VanillaOptimizer) # TODO


    from helpers import load_953_dataset, load_full_dataset
    data_directory = os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)), "data")
    training_datapoints, training_labels, testing_datapoints, testing_labels = load_953_dataset(data_directory)
    dataloader = npt.MNISTDataLoader(training_datapoints, training_labels, testing_datapoints, testing_labels)

    # the shape of the training examples might be a problem since they are (784,) right now. Not sure if they are handled correctly as column vectors.

    # make a pretty print function for the network.
    
    stochastic_steps = 500000
    result = train(network, dataloader, optimizer, stochastic_steps)

    testing_errors = result.testing_errors
    training_errors = result.training_errors
    corresponding_step = result.corresponding_step

    final_test_error_proportion = get_metrics(network, testing_datapoints, testing_labels)
    final_training_error_proportion = get_metrics(network, training_datapoints, training_labels)
    
    fig, ax = plt.subplots()
    ax.plot(corresponding_step, testing_errors, label='testing')
    ax.plot(corresponding_step, training_errors, label='training')
    ax.set(title=f'graph of classification error\nfinal testing error: {final_test_error_proportion}\nfinal training error: {final_training_error_proportion}', xlabel='stochastic step number', ylabel='classification error')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # **** requirements for data ****
    # data matricies must have columns as the datapoints, and the labels are a list of ints representing the class.
    # the ith column and the ith int in the labels go together. 
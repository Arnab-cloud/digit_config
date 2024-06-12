##### libraries
# Standard Library
import json
import sys
import random

#import my libraries
sys.path.append("../src")
import mnist_loader
import network2 as network


# 3rd_party libraries
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 5

def main():
    training()
    plotting()

def training():
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    results=[]
    net = network.Network([784, 30, 10])

    for eta in LEARNING_RATES:
        print("Training using the learning rate: ", eta)
        result = net.SGD(training_data, NUM_EPOCHS, 10, eta,
                               lmbda=5.0,
                               evaluation_data=validation_data,
                               monitor_training_cost=True)
        results.append(result)
        
    f = open("multiple_eta.json", "w")
    json.dump(results, f)
    f.close()

def plotting():
    # get the data from the file
    f = open("multiple_eta.json", "r")
    results = json.load(f)
    f.close()

    # initialize the plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for eta, result, color, in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label=f"eta: {eta}",
                color = color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cost")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
    
    
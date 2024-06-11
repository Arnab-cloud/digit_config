import network2 as network
import mnist_loader

net = network.Network([784, 30, 10])

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
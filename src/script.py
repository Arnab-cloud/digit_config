import network2 as network
import mnist_loader as mnist_loader


net = network.Network([784, 30, 10])

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
evaluation_cost, evaluation_accuracy, _, _, = net.SGD(training_data, epochs=30, mini_batch_size = 10, eta = 0.05, 
            lmbda=0.0,
            evaluation_data=test_data,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False)
net.save("net2")

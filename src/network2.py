## used cross_entropy cost function

#### Libraries
# Standard Libraries
import json
import random
import sys
import time

# 3rd_party libraries
import numpy as np


# define quadratic cost
class QuadraticCost(object):
    """Returns the function"""
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    
    """ Returns the delta or error using the function"""
    @staticmethod
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z)

class CrossEntropyCost(object):
    """Returns the function"""
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    """ Returns the delta or error using the function"""
    @staticmethod
    def delta(z, a, y):
        return (a-y)

# the main network class
class Network(object):
    """ Has the variables 'num_layers' => no. of hidden layers
        sizes = the list of the sizes of the hidden layers
        cost = cost class(type of the cost used)
        default_weight_initializer = a weight ans biases initializer
        when no specific method is used to initialize the weights and biases
    """
    def __init__(self,sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
    
    def default_weight_initializer(self):
        """Populate the wights and biases with a Gaussian distribution 
        with mean 0 and s.d = 1
        """
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(l,1) for l in self.sizes[1:]]

    def large_weight_initializer(self):
        """Populate the wights and biases with a Gaussian distribution 
        with mean 0 and s.d = 1
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(l,1) for l in self.sizes[1:]]

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if evaluation_data: n_test = len(evaluation_data)

        batch_size = len(training_data)

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # running multiple epochs
        for turn in range(epochs):
            time1 = time.time()
            random.shuffle(evaluation_data)

            mini_batches = [training_data[i:i+mini_batch_size] for i in range(0,batch_size,mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batches(mini_batch,eta, batch_size,lmbda)
            
            time2 = time.time()
            print("Epoch ", turn, " complete in ", time2-time1)

            if monitor_training_cost:
                cost = self.total_cost(training_data,lmbda)
                training_cost.append(cost)
                print("The training cost is ", cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("The training accuracy is ", accuracy,"/", batch_size)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data,lmbda)
                evaluation_cost.append(cost)
                print("The evaluation cost is ", cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("The evaluation accuracy is ", accuracy,"/", batch_size)
                    
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batches(self, mini_batch, eta, n, lmbda):
        mini_batch_size = len(mini_batch)
        sum_delta_w = [np.zeros(w.shape) for w in self.weights]
        sum_delta_b = [np.zeros(b.shape) for b in self.biases]
        # print("Before loop")
        # print("biases->{}, weights->{}".format(self.biases[0].shape, self.weights[0].shape))

        for x,y in mini_batch:
            nabla_w, nabla_b = self.backProp(x,y)
            # print("one done")
            sum_delta_w = [sw+nw for sw, nw in zip(sum_delta_w, nabla_w)]

            sum_delta_b = [sb+nb for sb, nb in zip(sum_delta_b, nabla_b)]
            
            self.biases = [b-(eta/mini_batch_size)*nb for b,nb in zip(self.biases, sum_delta_b)]

            # using regularized cost function
            self.weights = [(1.0 - eta*lmbda/n)*w-(eta/mini_batch_size)*nw for w,nw in zip(self.weights, sum_delta_w)]
            # print("biases->{}, weights->{}".format(self.biases[0].shape, self.weights[0].shape))
    
    def backProp(self,x,y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs=[]
        for w, b in zip(self.weights, self.biases):
            # print("w->{}, activation->{}, b->{}".format(w.shape, activation.shape, b.shape))
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_w,nabla_b)
    
    def feedForward(self,input_data):
        a = input_data
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a
    
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedForward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.feedForward(x)), y) for x, y in data]
        return sum(int(x==y) for x, y in results)


    def total_cost(self, data, lmbda, covert=False):
        """ "convert" is False if the "data" is the training data. 
        else "convert" is True
        """
        cost=0.0
        for x, y in data:
            a = self.feedForward(x)
            if(covert): y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        return cost + 0.5*(lmbda/len(data))*(sum(np.linalg.norm(w)**2 for w in self.weights))


    def evaluate(self, evaluation_data):
        sum=0
        for x,y in evaluation_data:
            if(np.argmax(self.feedForward(x)) == y):
                sum += 1
        return sum

    def save(self, filename):
        data={
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)
        }
        f = open(filename, "w")
        json.dump(data, filename)
        f.close()
    
# to load a network
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

# to vectorize the output y
def vectorized_result(y):
    ans = np.zeros((10,1))
    ans[y] = 1
    return ans

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def reLU(z):
    ans = z
    size = z.shape
    for i in range(size[0]):
        if z[i] <= 0:
            ans[i] = 0
    return ans
def reLU_prime(z):
    ans = z
    size = z.shape
    for i in range(size[0]):
        if z[i] <= 0:
            ans[i] = 0
        else:
            ans[i] =1
        return ans
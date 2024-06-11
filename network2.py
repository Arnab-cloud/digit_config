## used cross_entropy cost function

#### Libraries
# Standard Libraries
import random
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
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(l,1) for l in self.sizes[1:]]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)

        batch_size = len(training_data)
        for turn in range(epochs):
            time1 = time.time()
            random.shuffle(test_data)
            mini_batches = [training_data[i:i+mini_batch_size] for i in range(0,batch_size,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batches(mini_batch,eta)
            time2 = time.time()
            if test_data:
                print("Epoch {0}: {1}/{2}, took {3:.2f} seconds".format(turn, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {} complete in {3:.2f} seconds".format(turn, time2-time1))
        

    
    def evaluate(self, test_data):
        sum=0
        for x,y in test_data:
            if(np.argmax(self.feedForward(x)) == y):
                sum += 1
        return sum
    
    def feedForward(self,input_data):
        a = input_data
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def update_mini_batches(self, mini_batch, eta):
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

            self.weights = [w-(eta/mini_batch_size)*nw for w,nw in zip(self.weights, sum_delta_w)]
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
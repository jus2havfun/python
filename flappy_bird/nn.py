import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def normalize(X):
    return (X/np.max(X))

def load_data():
    mndata_train = MNIST('C:\\Users\\vikkyp.OPN\\Documents\\Nano Degree\\CourseEra\\Machine Learning\\mnist\\digits\\train', return_type='numpy')
    mndata_test = MNIST('C:\\Users\\vikkyp.OPN\\Documents\\Nano Degree\\CourseEra\\Machine Learning\\mnist\\digits\\test', return_type='numpy')
    Xtrain, Ytrain = mndata_train.load_training()
    Xtest, Ytest = mndata_test.load_testing()
    Xtrain = normalize(Xtrain)
    Xtest = normalize(Xtest)
    xtrain = [np.reshape(x, (784, 1)) for x in Xtrain]
    xtest = [np.reshape(x, (784, 1)) for x in Xtest]
    y = np.zeros((Ytrain.size, Ytrain.max()+1))
    y[np.arange(Ytrain.size), Ytrain] = 1.0
    ytrain = [np.reshape(y1, (10, 1)) for y1 in y]
    print("train and test data loaded")
    return (list(zip(xtrain, ytrain)), list(zip(xtest, Ytest)))


def sigmoid(x):
    return 1.0/(1.0+ np.exp(-x))

def sigmoid_prime(x):
    return (x * (1-x))

class neural_network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.numberOfLayers = len(sizes)

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x

    def train(self, data, epoch=10, learnrate=0.1, batchSize=None, testdata=None):
        n = len(data)
        #last_prediction=0.0
        if batchSize is None:
            batchSize = n
        for i in range(epoch):
            random.shuffle(data)
            batches = [data[k:k+batchSize] for k in range(0, n, batchSize)]
            for batch in batches:
                self.update_batch(batch, learnrate)
            if testdata:
                outputs = self.evaluate(testdata)
                prediction=(outputs/len(testdata))*100.0
                print ("Epoch {} complete with {} % correct".format(i, prediction))
                #if prediction > last_prediction:
                #    self.save()
            else:
                print("Epoch {} complete".format(i))

    def copy(self):
        nn = neural_network(self.sizes)
        nn.weights = self.weights
        nn.biases = self.biases
        return nn

    def update_batch(self, batch, learnrate):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            b, w = self.backpropogate(x, y)
            delta_b = [nb + dnb for nb, dnb in zip(delta_b, b)]
            delta_w = [nw + dnw for nw, dnw in zip(delta_w, w)]
        self.weights = [w - (learnrate/len(batch))*nw for w, nw in zip(self.weights, delta_w)]
        self.biases = [b - (learnrate/len(batch))*nb for b, nb in zip(self.biases, delta_b)]

    def backpropogate(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activations[-1] - y) * sigmoid_prime(activations[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        # now backpropogate through previous layers until we reach 
        # initial layer
        for l in range(2, self.numberOfLayers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(activations[-l])
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delta_b, delta_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def predict(self, x):
        test_results = [(np.argmax(self.feedforward(x)))]
        return test_results

    def eval(self, x):
        return self.feedforward(x)

    def mutate(self):
        mutation = np.random.normal(scale=1)
        self.biases = [b+mutation for b in self.biases]
        self.weights = [w+mutation for w in self.weights]

    def load(self):
        b, w=[],[]
        with open('nn.dat', 'rb') as f:
            for i in range(0, self.numberOfLayers-1):
                b.append(np.load(f))
                w.append(np.load(f))
        self.biases = b
        self.weights = w

    def save(self):
        biass, weighs=[],[]
        with open('nn.dat', 'wb') as f:
            for b, w in zip(self.biases, self.weights):
                np.save(f, b)
                np.save(f, w)
"""
if __name__=="__main__":
    training, testing = load_data()
    n_train, n_test = 0, 0
    if training is not None:
        n_train=len(training)
    if testing is not None:
        n_test=len(testing)
    nn = neural_network([784, 64, 32, 10])
    nn.load()
    nn.train(training, 30, 0.3, 10, testing)
    for j in range(10):
        i=random.randrange(n_test)
        s = str("Predicted digit:{}, expected:{}".format(nn.predict(testing[i][0]), testing[i][1]))
        plt.imshow(np.reshape(testing[i][0], (28,28)), cmap='gray_r')
        plt.text(0, -2, s)
        plt.show()
    nn.save()
"""

import numpy as np
import glob
from mnist import MNIST
from sklearn.model_selection import train_test_split
import pygame as py
from PIL import Image
import matplotlib.pyplot as plt
import random
import argparse
import time
import functools

OUTPUT_SIZE=10

def load_args():
    parser = argparse.ArgumentParser(description='Simple Neural Network with Backpropogation.')
    parser.add_argument("data_directory", default="./")
    parser.add_argument("--load_model_file", default="")
    parser.add_argument("--saved_images", default="")
    parser.add_argument("--hidden_units", nargs="*", type=int, default=[64, 32],)
    return parser.parse_args()

# normalize vector x between 0.0 and 1.0
def normalize(X):
    return (X/np.max(X))

def load_image(filename):
    img = py.image.load(filename)
    x = py.image.tostring(img, "RGB", True)
    img = Image.frombytes("RGB", img.get_size(), x).convert("L")
    x = img.tobytes('raw', 'L', 0, -1)
    X = [int(i) for i in x]
    return X


def load_files(path):
    d = {0:[], 1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    cnt = 0
    total_files=0
    for dir in glob.glob(path + "\*"):
        for file in glob.glob(dir +"\*"):
            d[cnt].append(file)
            total_files += 1
        cnt += 1

    A = np.empty([0, 784])
    B = np.empty([0, 1], dtype=np.int8)
    for key, value in d.items():
        for file in value:
            X = load_image(file)
            a = normalize(np.reshape(X, (784,)))
            A = np.vstack([A, a])
            B = np.vstack([B, key])
    B = B.reshape(total_files,)
    return A, B

# Below code load mnist train and test data.
# Returns: trainData, validationData and testingData sets.
# we split 10% of trainData as validationData.
def load_data(data_directory, saved_images):
    mndata_train = MNIST(data_directory + '\\train', return_type='numpy')
    mndata_test = MNIST(data_directory + '\\test', return_type='numpy')

    # Load training Data
    Xtrain, Ytrain = mndata_train.load_training()

    AImg, BImg = load_files(saved_images)

    Xtrain = np.concatenate((Xtrain, AImg), axis=0)
    Ytrain = np.concatenate((Ytrain, BImg), axis=0)

    # Load testing Data
    Xtest, Ytest = mndata_test.load_testing()
    
    # Normalize Xtrain and Xtest
    Xtrain = normalize(Xtrain)
    Xtest = normalize(Xtest)
    print (Xtrain.shape, type(Xtrain), Xtest.shape, type(Xtest))
    print (Xtrain[0].shape, type(Xtrain[0]), Xtest[0].shape, type(Xtest[0]))
    print (Ytrain.shape, type(Ytrain), Ytest.shape, type(Ytest))
    print (Ytrain[0].shape, type(Ytrain[0]))

    # Reshape Xtrain to be (781, 1) as inputs
    xtrain = [np.reshape(x, (784, 1)) for x in Xtrain]
    
    # create ValidationData as 10% of trainData 
    # to train our neural network
    tempList = list(zip(xtrain, Ytrain))

    # shuffle the data
    random.seed()
    random.shuffle(tempList)
    tempListLen = len(tempList)

    #split the data as trainData and ValidationData
    trainData, validationData = tempList[int(tempListLen*0.1):], tempList[:int(tempListLen*0.1)]

    # unpack the trainData to encode y as vector of 0's and 1's
    tempList = list(zip(*trainData))
    xtrain, Ytrain = tempList[0], np.asarray(tempList[1])

    # Convert Ytrain to be vector of 1's and 0's
    # if y = 1, then vector will be represented as [0 1 0 0 0 0 0 0 0 0]
    # if y = 5, then vector will be represented as [0 0 0 0 0 1 0 0 0 0]
    # if y = 7, then vector will be represented as [0 0 0 0 0 0 0 1 0 0]
    y = np.zeros((Ytrain.size, Ytrain.max()+1))
    y[np.arange(Ytrain.size), Ytrain] = 1.0
    ytrain = [np.reshape(y1, (OUTPUT_SIZE, 1)) for y1 in y]

    # Reshape Xtest to be (781, 1) as inputs
    xtest = [np.reshape(x, (784, 1)) for x in Xtest]
    print("train and test data loaded")
    trainData, testData = list(zip(xtrain, ytrain)), list(zip(xtest, Ytest))

    '''
    # Below code is inefficient way of splitting data into 
    # trainData and validationData, please see below.
    random.seed()
    random.shuffle(trainData)
    trainDataLen = len(trainData)
    trainData, validationData = trainData[int(trainDataLen*0.1):], trainData[:int(trainDataLen*0.1)]

    # Below code is acutally converting the vector y to digit
    # eg: y = [0 1 0 0 0 0 0 0 0 0] to 1
    # eg: y = [0 0 0 0 0 1 0 0 0 0] to 5
    # eg: y = [0 0 0 0 0 0 0 1 0 0] to 7
    xv, yv = [], []
    for x, y in validationData:
        xv.append(x)
        y = y.nonzero()[0][0]
        yv.append(y)
    yv = np.asarray(yv)
    return (trainData, list(zip(xv, yv)), testData)
    '''
    return trainData, validationData, testData


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return (x * (1-x))

class neural_network:
    def __init__(self, sizes):
        self.sizes = sizes
        '''
        if input sizes = [728, 128, 64, 32], then
               Weight Matrix    Bias Matrix
Input Layer      (128, 784)      (128, 1)
Hidden Layer 1   (64, 128)       (64, 1)
Hidden Layer 2   (32, 64)        (32, 1)
Output Layer     (10, 32)        (10, 1)
        '''
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #for w,b in zip(self.weights, self.biases):
        #    print(w.shape, b.shape)
        self.numberOfLayers = len(sizes)

    def copy(self):
        nn = neural_network(self.sizes)
        nn.weights = [np.copy(w) for w in self.weights]
        nn.biases = [np.copy(b) for b in self.biases]
        return nn

    def mutate(self):
        mutation = np.random.normal(scale=1)
        self.biases = [b+mutation for b in self.biases]
        self.weights = [w+mutation for w in self.weights]

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x)+b)
        return x

    def train(self, data, epoch=10, learnrate=0.1, batchSize=None, testdata=None):
        n = len(data)
        #last_prediction=0.0
        if batchSize is None:
            batchSize = n
        start = time.time()
        for i in range(epoch):
            random.shuffle(data)
            batches = [data[k:k+batchSize] for k in range(0, n, batchSize)]
            for batch in batches:
                self.update_batch(batch, learnrate)
            if testdata:
                outputs = self.evaluate(testdata)
                prediction=(outputs/len(testdata))*100.0
                end = time.time();
                print ("Epoch {} complete with {} % correct in {} secs".format(i, prediction, (end - start)))
                start = end;
                #if prediction > last_prediction:
                #    self.save()
            else:
                print("Epoch {} complete".format(i))

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
        x = self.feedforward(x)
        test_results = [(np.argmax(x))]
        return test_results

    def load(self, filename):
        b, w=[],[]
        try:
            with open(filename, 'rb') as f:
                old_sizes = np.load(f).tolist()
                if functools.reduce(lambda x, y : x and y, map(lambda p,q: p == q,old_sizes,self.sizes), True):
                    self.numberOfLayers = len(old_sizes)
                    for i in range(0, self.numberOfLayers-1):
                        b.append(np.load(f, allow_pickle=True))
                        w.append(np.load(f, allow_pickle=True))
                    self.biases = b
                    self.weights = w
                    print('model loaded with size: ', self.sizes)
                else:
                    print("Cannot load model file as provided sizes don't match. input:{}, loaded from file: {}".format(self.sizes, old_sizes))
        except:
            print ('File not found or invalid model file: {}'.format(filename))
            pass


    def save_model(self):
        self.save('nn_'+ ''.join([str(i) for i in self.sizes]) +'.dat')

    def save(self, filename):
        biass, weighs=[],[]
        with open(filename, 'wb') as f:
            np.save(f, np.array(self.sizes))
            for b, w in zip(self.biases, self.weights):
                np.save(f, b)
                np.save(f, w)

if __name__=="__main__":
    args= load_args()
    training, validation, testing = load_data(args.data_directory, args.saved_images)
    n_train, n_test = 0, 0
    if training is not None:
        n_train=len(training)
    if testing is not None:
        n_test=len(testing)
    sizes = [784]
    for x in args.hidden_units:
        sizes.append(x)
    sizes.append(OUTPUT_SIZE)
    print('train Size: {}, validation Size: {}, test Size: {}'.format(len(training), len(validation), len(testing)))
    nn = neural_network(sizes)
    if len(args.load_model_file) > 0:
        nn.load(args.load_model_file)
    nn.train(training, 1000, 0.1, 10, validation)
    correct, wrong = 0, 0
    for j in range(20):
        i=random.randrange(n_test)
        #print(type(testing[i][0]))
        #print(testing[i][0].shape)
        X, y = testing[i][0], testing[i][1]
        X = X.round()
        #print (len(X), X.shape, X)
        p = nn.predict(X)
        if p == y:
            correct += 1
        else:
            wrong += 1    
        s = str("Predicted digit:{}, expected:{}, accuracy:{}".format(p, y, correct/(correct+wrong+0.0001)))
        plt.imshow(np.reshape(X, (28,28)), cmap='gray_r')
        X = np.reshape(X, (784,))
        for a in range(28):
            for b in range(28):
                if (X[a*28 + b] > 0):
                    print ('1', end=' ')
                else:
                    print ('0', end=' ')
            print ()
        plt.text(0, -2, s)
        plt.show()
    sizes_str="_".join([str(i) for i in nn.sizes])
    nn.save('nn_'+ args.data_directory[args.data_directory.rfind("\\")+1:] + '_' + sizes_str +'.dat')

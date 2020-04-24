import numpy as np
import time
from scipy.special import expit
from scipy.special import logsumexp
import sys

BATCH_SIZE = 128
LEARNING = 0.1

def sigmoid_derivative(x):
    s = expit(x)
    dx = s * (1 - s)
    return dx


def relu(x):
   return np.maximum(0,x)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class BaseData:
    def __init__(self):
        self.activation1 = list()
        self.activation2 = list()
        self.output = list()
        self.label = None
        self.A = list()
        return

    def transform_label(self, label):
        label_t = np.zeros((label.shape[0], 10))
        for i in range(len(label)):
            j = label[i]
            j = int(j)
            label_t[i, j] = 1
        self.label = label_t.reshape((len(label), 10))

    def clear_cache(self):
        self.activation1 = list()
        self.activation2 = list()
        self.output = list()


class Trainer(BaseData):
    def __init__(self, image_name, label_name):
        super().__init__()
        train_data = np.genfromtxt(image_name, delimiter=',')
        train_data = np.reshape(train_data, [len(train_data), 784])
        self.data = train_data[0:10000, :] / 255
        train_label = np.genfromtxt(label_name, delimiter=',')
        self.transform_label(train_label[0:10000])
        return

class Test(BaseData):
    def __init__(self, image_name, label_name=None):
        super().__init__()
        test_data = np.genfromtxt(image_name, delimiter=',')
        test_data = np.reshape(test_data, [len(test_data), 784])
        self.data = test_data / 255
        if label_name:
            test_label = np.genfromtxt(label_name, delimiter=',')
            self.transform_label(test_label)
        return

    def output_csv(self):
        idx = np.zeros((len(self.output), 1))
        for i in range(len(self.output)):
            idx[i] = np.argmax(self.output[i])
        np.savetxt('test_predictions.csv', idx.astype(int), fmt='%i', delimiter=',')
        return

class NeuralNet:
    def __init__(self, train_image, train_label, test_image):
        self.train_image = train_image
        self.train_label = train_label
        self.test_image = test_image

        self.w1 = np.random.randn(784, 128) * np.sqrt(1 / 784)
        self.w2 = np.random.randn(128, 32) * np.sqrt(1 / 128)
        self.w3 = np.random.randn(32, 10) * np.sqrt(1 / 32)
        self.b1 = np.zeros((1, 128))
        self.b2 = np.zeros((1, 32))
        self.b3 = np.zeros((1, 10))
        return

    def feed_forward(self, data, obj):
        obj.clear_cache()
        for i in range(data.shape[0]):
            act1 = relu(np.matmul(data[i], self.w1) + self.b1)
            act2 = relu(np.matmul(act1, self.w2) + self.b2)
            op = softmax(np.matmul(act2, self.w3) + self.b3)
            obj.activation1.append(act1)
            obj.activation2.append(act2)
            obj.output.append(op)
        return

    # get accuracy
    def get_accuracy(self, label, obj, refresh=False):
        total = 0
        match = 0
        for i in range(len(label)):
            predict_max = np.argmax(obj.output[i])
            label_max = np.argmax(label[i])
            if predict_max == label_max:
                match += 1
            total += 1
        accuracy = (float(match) / total) * 100
        if refresh:
            obj.A = list()
        obj.A.append(accuracy)

    def backward_propagation(self, data, label, obj):
        w1_update = list()
        w2_update = list()
        w3_update = list()
        b1_update = list()
        b2_update = list()
        b3_update = list()

        for i in range(data.shape[0]):
            data_origin = np.reshape(data[i], [1, 784])
            output_error = obj.output[i] - label[i]
            w3_delta = np.matmul(obj.activation2[i].T, output_error)
            b3_delta = output_error
            activation2_error = np.multiply(np.matmul(output_error, self.w3.T), relu_derivative(obj.activation2[i]))
            w2_delta = np.matmul(obj.activation1[i].T, activation2_error)
            b2_delta = activation2_error
            activation1_error = np.multiply(np.matmul(activation2_error, self.w2.T), relu_derivative(obj.activation1[i]))
            w1_delta = np.matmul(data_origin.T, activation1_error)
            b1_delta = activation1_error
            w1_update.append(w1_delta)
            w2_update.append(w2_delta)
            w3_update.append(w3_delta)
            b1_update.append(b1_delta)
            b2_update.append(b2_delta)
            b3_update.append(b3_delta)

        self.w1 = self.w1 - LEARNING * sum(w1_update) / len(w1_update)
        self.w2 = self.w2 - LEARNING * sum(w2_update) / len(w2_update)
        self.w3 = self.w3 - LEARNING * sum(w3_update) / len(w3_update)
        self.b1 = self.b1 - LEARNING * sum(b1_update) / len(b1_update)
        self.b2 = self.b2 - LEARNING * sum(b2_update) / len(b2_update)
        self.b3 = self.b3 - LEARNING * sum(b3_update) / len(b3_update)

    def train(self):
        self.trainer = Trainer(self.train_image, self.train_label)
        if self.trainer.data.shape[0] < 30000:
            NUM_EPOCH = 20
        else:
            NUM_EPOCH = 10

        num_batch = self.trainer.data.shape[0] // BATCH_SIZE
        for ep in range(NUM_EPOCH):
            for i in range(num_batch + 1):
                if i == BATCH_SIZE:
                    start = (BATCH_SIZE * i) % self.trainer.data.shape[0]
                    end = self.trainer.data.shape[0] - 1
                else:
                    start = (BATCH_SIZE * i) % self.trainer.data.shape[0]
                    end = start + BATCH_SIZE
                batch_data = self.trainer.data[start:end, :]
                self.feed_forward(batch_data, self.trainer)
                batch_label = self.trainer.label[start:end, :]
                self.backward_propagation(batch_data, batch_label, self.trainer)
        self.feed_forward(self.trainer.data, self.trainer)
        self.get_accuracy(self.trainer.label, self.trainer, refresh=True)
        print(self.trainer.A)
        return

    def get_result(self):
        self.test = Test(self.test_image, 'test_label.csv')
        self.feed_forward(self.test.data, self.test)
        self.test.output_csv()
        self.get_accuracy(self.test.label, self.test)
        print(self.test.A)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(1)
    a = time.time()
    train_image = sys.argv[1]
    train_label = sys.argv[2]
    test_image = sys.argv[3]
    x = NeuralNet(train_image, train_label, test_image)
    x.train()
    x.get_result()
    b = time.time()
    print(b - a)

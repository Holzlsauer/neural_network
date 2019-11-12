import os
import numpy as np
import pandas as pd
from scipy.special import expit


class NeuralNetwork():
    """Neural network designed to classif the MNIST dataset"""
    def __init__(self, images, labels):
        img = np.asarray(images)
        self.images = img.reshape(img.shape[0],img.shape[1]*img.shape[2])*255
        self.bkp_labels = np.asarray(labels)
        self.labels = self.treat_labels(self.bkp_labels)
        # 1 hidden leyer, 16 node
        """
        a weight matrix of shape (n, m) where n is the number of output neurons (neurons in the next layer) and m is the number of input neurons (neurons in the previous layer)
        """
        self.weights_1 = np.random.rand(16, self.images.shape[1])
        self.weights_2 = np.random.rand(10, 16)
        # Output same number of images, 10 scores
        self.output = np.zeros((self.images.shape[0], 10))

    @staticmethod
    def sigmoid(x, der=False):
        """Activation function"""
        s = expit(x)
        """
        The expit function, also known as the logistic function, is defined as
        expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.
        - SciPy documentation
        """
        if der:
            return x*(1-x)
        return s

    def treat_labels(self, label):
        """Adjust the shape of labels"""
        label = np.asarray(label)
        if not label.shape == (label.shape[0], 10):
            tmp_output = np.zeros((label.shape[0],10))
            for k, output in enumerate(label):
                """
                A ideia aqui é criar a saída como com arrays de 10 espaços,
                com a valor 1 preenchido na saída correta e 0 nas demais
                """
                tmp_output[k][output] = 1
            return tmp_output
        else:
            return label

    def predict(self, x_test, y_test):
        """Compute the output"""
        x_test = np.asarray(x_test)
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])*255
        y_test = np.asarray(y_test)
        # Prediction part
        hidden = self.sigmoid(np.dot(x_test, self.weights_1.T))
        output = self.sigmoid(np.dot(hidden, self.weights_2.T))
        percentage = 0
        for k, prediction in enumerate(output):
            #print(np.argmax(prediction), self.lbl[k])
            if np.argmax(prediction) == y_test[k]:
                percentage += 1
        print(f'Porcentagem de acerto: {percentage*100/len(y_test)}%')
        return output

    def feed_forward(self):
        """Compute the output"""
        self.hidden = self.sigmoid(np.dot(self.images, self.weights_1.T))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_2.T))

    def back_propagation(self):
        """Adjust the weights by the loss gradient"""
        error = 2 * (self.labels - self.output) # -1 * (labels - output)
        # Gradient of weights 2
        grad_2 = error * self.sigmoid(self.output, der=True)
        grad_2 = np.dot(grad_2.T, self.hidden)
        # Gradient of weights 1
        grad_1 = error * self.sigmoid(self.output, der=True)
        grad_1 = np.dot(grad_1, self.weights_2)
        grad_1 = grad_1 * self.sigmoid(self.hidden, der=True)
        grad_1 = np.dot(grad_1.T, self.images)
        # Update the weights
        self.weights_1 += grad_1
        self.weights_2 += grad_2

    def save_state(self):
        """Save trained parameters"""
        with open ('state.txt', 'w') as state:
            state.write('1500 iteracoes')
        pd.DataFrame(self.weights_1).to_csv('weights_1.csv')
        pd.DataFrame(self.weights_2).to_csv('weights_2.csv')

    def load_state(self, folder=None):
        """Load trained parameters"""
        if folder is None:
            self.weights_1 = pd.read_csv('weights_1.csv', index_col=0).values
            self.weights_2 = pd.read_csv('weights_2.csv', index_col=0).values
        else:
            try:
                path = os.pardir + f'/{folder}/'
                w1 = pd.read_csv(f'{path}weights_1.csv', index_col=0).values
                w2 = pd.read_csv(f'{path}weights_2.csv', index_col=0).values
                self.weights_1 = w1
                self.weights_2 = w2
            except Exception as e:
                print(e)
                exit()

    def train(self, itr=1500):
        """Train the model"""
        for _ in range(itr):
            self.feed_forward()
            self.back_propagation()


if __name__ == '__main__':
    pass
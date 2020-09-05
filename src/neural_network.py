import os
import numpy as np
import pandas as pd
from scipy.special import expit


class NeuralNetwork():
    """Neural network designed to classif the MNIST dataset"""
    def __init__(self, images, labels):
        img = np.asarray(images)
        # Reshape images from 2D array to 1D
        self.images = img.reshape(img.shape[0],img.shape[1]*img.shape[2])*255
        self.backup_labels = np.asarray(labels)
        self.labels = self.treat_labels(self.backup_labels)
        self.weights_layer_1 = np.random.rand(self.images.shape[1], 16)
        self.weights_layer_2 = np.random.rand(16, 10)
        # Output same number of images, 10 scores
        self.output = np.zeros((self.images.shape[0], 10))

    @staticmethod
    def sigmoid(x, derivative=False):
        """Activation function"""
        s = expit(x)
        """
        The expit function, also known as the logistic function, is defined as
        expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.
        - SciPy documentation
        """
        if derivative:
            return s*(1-s)
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
        hidden = self.sigmoid(np.dot(x_test, self.weights_layer_1))
        output = self.sigmoid(np.dot(hidden, self.weights_layer_2))
        right_predicition = 0
        for k, prediction in enumerate(output):
            if np.argmax(prediction) == y_test[k]:
                right_predicition += 1
        print(f'Porcentagem de acerto: {right_predicition*100/len(y_test)}%')
        with open('progress', 'a') as f:
            f.write(f'Porcentagem de acerto: {right_predicition*100/len(y_test)}%')
        return output

    def feed_forward(self):
        """Compute the output"""
        self.hidden = self.sigmoid(np.dot(self.images, self.weights_layer_1))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_layer_2))

    def back_propagation(self):
        """Adjust the weights by the loss gradient"""
        error = (self.labels - self.output) # -1 * (labels - output)
        # Gradient descent of cost function in terms of weights_layer_2
        g2 = error*self.sigmoid(self.output, derivative=True)
        g2 = np.dot(self.hidden.T, g2)
        # Gradient descent of cost function in terms of weights_layer_1
        g1 = error*self.sigmoid(self.output, derivative=True)
        g1 = np.dot(g1, self.weights_layer_2.T)
        g1 = g1*self.sigmoid(self.hidden, derivative=True)
        g1 = np.dot(self.images.T, g1)
        # Update values
        self.weights_layer_1 += g1
        self.weights_layer_2 += g2

    def train(self, itr=1500):
        """Train the model"""
        for _ in range(itr):
            self.feed_forward()
            self.back_propagation()


if __name__ == '__main__':
    pass
import src.mnist_loader as mnist
import src.neural_network as nn


def main():
    neural = nn.NeuralNetwork(mnist.train_images(), mnist.train_labels())
    neural.train()
    neural.predict(mnist.test_images(), mnist.test_labels())

if __name__ == '__main__':
    main()
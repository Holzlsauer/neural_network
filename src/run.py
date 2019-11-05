import mnist_loader as mnist
import neural_network as nn


def main():
    neural = nn.NeuralNetwork(mnist.train_images(), mnist.train_labels())
    neural.load_state()
    neural.predict(mnist.test_images(), mnist.test_labels())

if __name__ == '__main__':
    main()
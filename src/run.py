import mnist_loader as mnist
import neural_network as nn


def main():
    neural = nn.NeuralNetwork(mnist.train_images(), mnist.train_labels())
    neural.load_state()
    for _ in range(100):
        neural.train(itr=100)
        neural.predict(mnist.test_images(), mnist.test_labels())
    neural.save_state()

if __name__ == '__main__':
    main()
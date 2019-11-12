import matplotlib.pyplot as plt

def main():
    with open('progress', 'r') as f:
        progress = [float(pct[-7:-2].strip()) for pct in f.readlines()]
    iterations = [(i+1)*100 for i in range(len(progress))]
    plt.plot(iterations, progress)
    plt.show()

if __name__ == '__main__':
    main()
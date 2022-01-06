import matplotlib.pyplot as plt


def plot_loss(train_loss, output_dir):
    x = range(1, len(train_loss) + 1)
    plt.plot(x, train_loss, '.-')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(output_dir + 'train_loss.svg')


if __name__ == "__main__":
    plot_loss(0, 0)

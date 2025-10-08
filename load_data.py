import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_show_samples():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("x_train:", x_train.shape, "x_test:", x_test.shape)
    plt.figure(figsize=(8, 6))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_train[i])
        plt.title(class_names[y_train[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return (x_train, y_train), (x_test, y_test)

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

#10 classes du dataset CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_show_samples():
    #Chargement du dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalisation des images
    print("x_train:", x_train.shape, "x_test:", x_test.shape)

    # Figure 8x6
    plt.figure(figsize=(8, 6))
    # Boucle pour afficher 12 échantillons d'images
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_train[i])
        plt.title(class_names[y_train[i]])
        #Cacher les axes
        plt.axis('off')
    # Ajustement des espacements
    plt.tight_layout()
    # Affichage de la figure
    plt.show()
    return (x_train, y_train), (x_test, y_test)

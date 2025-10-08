import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Définition des noms des classes du dataset CIFAR-10
# Ces 10 classes correspondent aux 10 catégories d'objets dans CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_show_samples():
    """
    Fonction qui charge le dataset CIFAR-10 et affiche un échantillon d'images
    
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
               - x_train: images d'entraînement (50000, 32, 32, 3)
               - y_train: labels d'entraînement (50000, 1)
               - x_test: images de test (10000, 32, 32, 3)
               - y_test: labels de test (10000, 1)
    """
    # Chargement du dataset CIFAR-10 depuis TensorFlow/Keras
    # Le dataset est automatiquement divisé en train/test
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Affichage des dimensions des données pour vérification
    print("x_train:", x_train.shape, "x_test:", x_test.shape)
    
    # Création d'une figure matplotlib de taille 8x6 pouces
    plt.figure(figsize=(8, 6))
    
    # Boucle pour afficher 12 échantillons d'images
    for i in range(12):
        # Création d'un subplot dans une grille 3x4 (3 lignes, 4 colonnes)
        plt.subplot(3, 4, i + 1)
        
        # Affichage de l'image i depuis le dataset d'entraînement
        plt.imshow(x_train[i])
        
        # Définition du titre avec le nom de la classe correspondante
        # y_train[i][0] car y_train[i] est un array contenant l'index de classe
        plt.title(class_names[y_train[i][0]])
        
        # Suppression des axes pour un affichage plus propre
        plt.axis('off')
    
    # Ajustement automatique de l'espacement entre les subplots
    plt.tight_layout()
    
    # Affichage de la figure
    plt.show()
    
    # Retour des données chargées pour utilisation ultérieure
    return (x_train, y_train), (x_test, y_test)
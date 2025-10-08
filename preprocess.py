import numpy as np

def normalize_data(x_train, x_test):
    #Affichage des valeurs min et max avant normalisation
    print("Avant:", x_train.min(), x_train.max())
    #Conversiton en flot32
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    #Affichage des valeurs min et max apres normalisation
    print("Après:", x_train.min(), x_train.max())
    return x_train, x_test

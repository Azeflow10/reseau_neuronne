import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from load_data import class_names

def evaluate_and_predict(x_test, y_test):\

    #Chargement du meilleur modele sauvegarder perdant l'entrainement 

    model = load_model("best_model.h5")

    #Evaluation du modele sur les donnees de test
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    #Prediction sur les 12 premiere immages de test

    preds = model.predict(x_test[12])
    pred_labs = np.argmax(preds, axis=1)

    #Affichage des resultats de prediction

    plt.figure(figsize=(8,6))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(x_test[i])
        plt.title(f"Pred: {class_names[pred_labs[i][0]]}\nTrue: {class_names[y_test[i][0]]}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
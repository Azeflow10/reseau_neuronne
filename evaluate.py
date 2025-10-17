import numpy as np
import matplotlib.pyplot as plt
from load_data import class_names


def evaluate_and_predict(model, x_test, y_test):
    
    # Évaluation globale
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Prédictions sur les 12 premières images
    k = min(12, len(x_test))
    preds = model.predict(x_test[:k], verbose=0)
    pred_labels = np.argmax(preds, axis=1) 

    # Affichage des résultats
    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[i])
        # y_test peut être (N,1) -> extraire l'entier
        true_idx = int(y_test[i]) if np.ndim(y_test[i]) == 0 else int(y_test[i][0])
        plt.title(f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt

def plot_training(history):
    #Extraction des donnees de training
    h = history.history

    #Creation de la liste des epochs
    epochs = range(1, len(h['loss']) + 1)

    #Plot des courbes de loss et accuracy

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, h['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, h['val_loss'], 'ro-', label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')  

    # Plot accuracy

    plt.subplot(1, 2, 2)
    plt.plot(epochs, h['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, h['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Affichage des courbes

    plt.tight_layout()
    plt.show()
from tensorflow.keras import layers, models


def make_simple_cnn():
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),

        # Reduction spatiale par sous‑echantillonnage
        layers.MaxPooling2D((2, 2)),

        #extraction de caractéristiques plus riches avec plus de filtres
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

        # Nouvelle reduction spatiale
        layers.MaxPooling2D((2, 2)),

        # Passage du tenseur 3D 
        layers.Flatten(),

        # Couche entierement connectee pour la combinaison des caracteristiques
        layers.Dense(64, activation='relu'),

        # Couche de sortie: 10 probabilités (une par classe), somme = 1
        layers.Dense(10, activation='softmax'),
    ])
    model.summary()
    return model

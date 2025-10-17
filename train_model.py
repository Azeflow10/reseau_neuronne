import tensorflow as tf

def train_model(model, x_train, y_train):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        callbacks=callbacks
    )

    model.save("final_model.h5")
    return history

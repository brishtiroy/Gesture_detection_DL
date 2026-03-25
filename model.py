import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_lstm(num_classes):
    model = models.Sequential()

    # CNN feature extractor (TimeDistributed)
    model.add(layers.TimeDistributed(
        layers.Conv2D(32, (3,3), activation='relu'),
        input_shape=(10, 64, 64, 3)
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(2,2)))

    model.add(layers.TimeDistributed(
        layers.Conv2D(64, (3,3), activation='relu')
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(2,2)))

    model.add(layers.TimeDistributed(layers.Flatten()))

    # LSTM for sequence
    model.add(layers.LSTM(64))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
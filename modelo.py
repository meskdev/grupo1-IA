import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, callbacks


def carregar_dados():
    (treino, teste), info = tfds.load('emnist/byclass',
                                      split=['train', 'test'],
                                      as_supervised=True,
                                      with_info=True)

    def preprocessar(imagem, rotulo):
        imagem = tf.transpose(imagem, [1, 0, 2])
        imagem = tf.image.flip_left_right(imagem)
        return tf.cast(imagem, tf.float32) / 255.0, tf.one_hot(rotulo, 62)

    return treino.map(preprocessar).batch(128), teste.map(preprocessar).batch(128)


def criar_modelo():
    modelo = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(62, activation='softmax')
    ])

    modelo.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo


def treinar():
    treino, teste = carregar_dados()
    modelo = criar_modelo()

    historico = modelo.fit(
        treino,
        validation_data=teste,
        epochs=30,
        callbacks=[
            callbacks.EarlyStopping(patience=5),
            callbacks.ReduceLROnPlateau(patience=3)
        ]
    )

    modelo.save('emnist_model.h5')
    print("Modelo treinado e salvo!")
    return historico


if __name__ == '__main__':
    treinar()

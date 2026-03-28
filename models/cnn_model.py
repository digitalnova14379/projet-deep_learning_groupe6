"""
Mission 1 - Modèle CNN pour la classification d'images CIFAR-10
Auteur : ABTELX / ENSPD Deep Learning
Architecture : API Subclassing de Keras (tf.keras.Model)
"""

import tensorflow as tf
from tensorflow.keras import layers


class CustomCNN(tf.keras.Model):
    """
    Réseau de neurones convolutif personnalisé pour CIFAR-10.

    Architecture :
        - Data Augmentation (RandomFlip, RandomRotation)
        - 3 blocs Conv2D + BatchNormalization + MaxPooling2D
        - Flatten + Dense + Dropout (classifieur MLP)
        - Couche de sortie : Dense(10, softmax)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Initialise les couches du modèle.

        Args:
            num_classes  : Nombre de classes (10 pour CIFAR-10).
            dropout_rate : Taux de Dropout pour la régularisation.
        """
        super(CustomCNN, self).__init__()

        # ── Data Augmentation (intégrée au modèle) ──────────────────────────
        self.augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="data_augmentation")

        # ── Bloc 1 : Conv2D 32 filtres ───────────────────────────────────────
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                   name="conv1")
        self.bn1   = layers.BatchNormalization(name="bn1")
        self.pool1 = layers.MaxPooling2D((2, 2), name="pool1")

        # ── Bloc 2 : Conv2D 64 filtres ───────────────────────────────────────
        self.conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                   name="conv2")
        self.bn2   = layers.BatchNormalization(name="bn2")
        self.pool2 = layers.MaxPooling2D((2, 2), name="pool2")

        # ── Bloc 3 : Conv2D 128 filtres ──────────────────────────────────────
        self.conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                   name="conv3")
        self.bn3   = layers.BatchNormalization(name="bn3")
        self.pool3 = layers.MaxPooling2D((2, 2), name="pool3")

        # ── Classifieur MLP ───────────────────────────────────────────────────
        self.flatten  = layers.Flatten(name="flatten")
        self.dense1   = layers.Dense(256, activation='relu', name="dense1")
        self.dropout  = layers.Dropout(dropout_rate, name="dropout")
        self.dense2   = layers.Dense(128, activation='relu', name="dense2")
        self.output_layer = layers.Dense(num_classes, activation='softmax',
                                         name="output")

    def call(self, inputs, training: bool = False):
        """
        Passe forward du réseau.

        Args:
            inputs   : Tenseur d'images (batch, 32, 32, 3), valeurs dans [0, 1].
            training : Booléen — active/désactive Dropout et BatchNorm en mode entraînement.

        Returns:
            Tenseur de probabilités (batch, num_classes).
        """
        x = self.augmentation(inputs, training=training)

        # Bloc 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        # Bloc 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        # Bloc 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        # Classifieur
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)

    def get_config(self):
        return {"num_classes": 10, "dropout_rate": 0.5}

    @classmethod
    def from_config(cls, config):
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(**config)

    def build_graph(self, input_shape=(32, 32, 3)):
        """
        Force la construction du graphe pour afficher model.summary().

        Args:
            input_shape : Forme d'une image sans la dimension batch.
        """
        inputs = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
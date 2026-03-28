"""
Mission 2 - Modèle LSTM pour la prédiction de séries temporelles
Auteur : ABTELX / ENSPD Deep Learning
Architecture : API Subclassing de Keras (tf.keras.Model)
Dataset : Météo de Jena (température T+1 step prédiction)
"""

import tensorflow as tf
from tensorflow.keras import layers


class CustomLSTM(tf.keras.Model):
    """
    Réseau LSTM pour la prédiction du prochain pas de temps (T+1).

    Architecture :
        - LSTM(128, return_sequences=True)  — capture les dépendances longues
        - Dropout(0.2)
        - LSTM(64, return_sequences=False)  — dernier état caché uniquement
        - Dropout(0.2)
        - Dense(32, relu)
        - Dense(1)                          — prédiction scalaire T+1
    """

    def __init__(self, dropout_rate: float = 0.2):
        """
        Initialise les couches du modèle.

        Args:
            dropout_rate : Taux de Dropout pour la régularisation entre les couches LSTM.
        """
        super(CustomLSTM, self).__init__()

        # ── Couches LSTM ─────────────────────────────────────────────────────
        self.lstm1   = layers.LSTM(128, return_sequences=True, name="lstm1")
        self.drop1   = layers.Dropout(dropout_rate, name="dropout1")

        # return_sequences=False : on ne récupère que le dernier état caché
        self.lstm2   = layers.LSTM(64, return_sequences=False, name="lstm2")
        self.drop2   = layers.Dropout(dropout_rate, name="dropout2")

        # ── Tête de régression ────────────────────────────────────────────────
        self.dense1       = layers.Dense(32, activation='relu', name="dense1")
        self.output_layer = layers.Dense(1, name="output")  # Prédiction T+1

    def call(self, inputs, training: bool = False):
        """
        Passe forward du réseau.

        Args:
            inputs   : Tenseur (batch, sequence_length, features).
            training : Booléen — active/désactive Dropout.

        Returns:
            Tenseur de prédiction scalaire (batch, 1).
        """
        x = self.lstm1(inputs)
        x = self.drop1(x, training=training)
        x = self.lstm2(x)
        x = self.drop2(x, training=training)
        x = self.dense1(x)
        return self.output_layer(x)

    def get_config(self):
        return {"dropout_rate": 0.2}

    @classmethod
    def from_config(cls, config):
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(**config)

    def build_graph(self, sequence_length: int, num_features: int):
        """
        Force la construction du graphe pour afficher model.summary().

        Args:
            sequence_length : Longueur de la fenêtre temporelle.
            num_features    : Nombre de variables d'entrée.
        """
        inputs = tf.keras.Input(shape=(sequence_length, num_features))
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
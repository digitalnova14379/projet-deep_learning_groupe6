"""
Utilitaires de chargement et préparation des données.
Auteur : ABTELX / ENSPD Deep Learning

Contenu :
    - load_cifar10()          : Chargement et pipeline tf.data pour CIFAR-10
    - load_jena_climate()     : Téléchargement et préparation du dataset Météo Jena
    - build_lstm_dataset()    : Construction des fenêtres temporelles (sliding window)
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION 1 — CIFAR-10
# ═══════════════════════════════════════════════════════════════════════════════

def load_cifar10(batch_size: int = 64) -> tuple:
    """
    Charge CIFAR-10, normalise les pixels dans [0, 1] et construit
    des pipelines tf.data.Dataset optimisés pour l'entraînement.

    Args:
        batch_size : Taille des mini-lots (défaut : 64).

    Returns:
        train_ds : tf.data.Dataset — données d'entraînement (augmentation incluse via le modèle).
        val_ds   : tf.data.Dataset — données de validation.
        test_ds  : tf.data.Dataset — données de test.
    """
    # Chargement depuis Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation des pixels : uint8 [0, 255] → float32 [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Aplatissement des labels : (N, 1) → (N,)
    y_train = y_train.flatten()
    y_test  = y_test.flatten()

    # Split train / validation (80% / 20%)
    val_split = int(0.2 * len(x_train))
    x_val, y_val     = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]

    # ── Constructeur de pipeline tf.data ────────────────────────────────────
    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(x, y, shuffle: bool = False) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x), seed=42)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    train_ds = make_dataset(x_train, y_train, shuffle=True)
    val_ds   = make_dataset(x_val,   y_val)
    test_ds  = make_dataset(x_test,  y_test)

    print(f"[CIFAR-10] Train : {len(x_train):,} | Val : {len(x_val):,} | Test : {len(x_test):,}")
    print(f"[CIFAR-10] Batch size : {batch_size} | Batches/époque : {len(train_ds)}")

    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION 2 — DATASET MÉTÉO JENA
# ═══════════════════════════════════════════════════════════════════════════════

def load_jena_climate(data_dir: str = "data/") -> tuple:
    """
    Télécharge (si nécessaire) et charge le dataset Météo de Jena.
    Retourne les données brutes de température et le scaler ajusté.

    Args:
        data_dir : Dossier local où stocker le fichier CSV.

    Returns:
        data_scaled : np.ndarray — données normalisées dans [0, 1].
        scaler      : MinMaxScaler ajusté (pour inverse_transform après prédiction).
        df          : pd.DataFrame — données brutes (utile pour visualisation).
    """
    import pandas as pd

    csv_path = os.path.join(data_dir, "jena_climate_2009_2016.csv")

    # ── Téléchargement si absent ─────────────────────────────────────────────
    if not os.path.exists(csv_path):
        print("[Jena] Téléchargement du dataset en cours...")
        url = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
               "jena_climate_2009_2016.csv.zip")
        import shutil, glob
        # Keras télécharge dans son cache (~/.keras/datasets/)
        keras_cache = tf.keras.utils.get_file(
            fname="jena_climate_2009_2016.csv.zip",
            origin=url,
            extract=True,
        )
        # Recherche le CSV dans tout le cache Keras (Keras peut créer un sous-dossier _extracted)
        keras_datasets_dir = os.path.dirname(keras_cache)
        matches = glob.glob(
            os.path.join(keras_datasets_dir, "**", "jena_climate_2009_2016.csv"),
            recursive=True
        )
        if not matches:
            raise FileNotFoundError(
                f"CSV Jena introuvable dans {keras_datasets_dir}. "
                "Vérifiez votre connexion et relancez."
            )
        keras_csv = matches[0]
        # Copie vers data/ pour le retrouver à chaque lancement
        os.makedirs(data_dir, exist_ok=True)
        shutil.copy(keras_csv, csv_path)
        print(f"[Jena] Dataset copié vers : {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[Jena] Dimensions brutes : {df.shape}")

    # ── Sous-échantillonnage : 1 mesure / heure (données toutes les 10 min) ──
    df = df[5::6].reset_index(drop=True)
    print(f"[Jena] Après sous-échantillonnage (1/h) : {df.shape}")

    # ── Sélection de la colonne cible : T (degC) ─────────────────────────────
    temperature = df[["T (degC)"]].values.astype("float32")

    # ── Normalisation MinMax ─────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(temperature)

    return data_scaled, scaler, df


def build_lstm_dataset(
    data_scaled: np.ndarray,
    sequence_length: int = 24,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> tuple:
    """
    Transforme une série temporelle 1D en fenêtres glissantes (sliding window)
    et construit des pipelines tf.data.Dataset prêts pour l'entraînement LSTM.

    Args:
        data_scaled     : Série normalisée (N, 1).
        sequence_length : Nombre de pas de temps en entrée (lookback window).
        batch_size      : Taille des mini-lots.
        train_ratio     : Proportion des données pour l'entraînement.
        val_ratio       : Proportion pour la validation.

    Returns:
        train_ds       : tf.data.Dataset d'entraînement.
        val_ds         : tf.data.Dataset de validation.
        test_ds        : tf.data.Dataset de test.
        y_test_inverse : np.ndarray — vraies valeurs dé-normalisées (pour visualisation).
    """
    n = len(data_scaled)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_data = data_scaled[:train_end]
    val_data   = data_scaled[train_end:val_end]
    test_data  = data_scaled[val_end:]

    AUTOTUNE = tf.data.AUTOTUNE

    def make_timeseries_ds(data: np.ndarray, shuffle: bool = False) -> tf.data.Dataset:
        """Crée un dataset de fenêtres glissantes via timeseries_dataset_from_array."""
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data[:-1],           # features : tous sauf le dernier
            targets=data[sequence_length:],  # cible : T+1
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=42
        )
        return ds.prefetch(AUTOTUNE)

    train_ds = make_timeseries_ds(train_data, shuffle=True)
    val_ds   = make_timeseries_ds(val_data)
    test_ds  = make_timeseries_ds(test_data)

    # Vraies valeurs de test dé-normalisées pour le graphique final
    y_test_raw = test_data[sequence_length:]

    print(f"[LSTM] Sequence length : {sequence_length} | Batch size : {batch_size}")
    print(f"[LSTM] Train : {len(train_data):,} | Val : {len(val_data):,} | Test : {len(test_data):,}")

    return train_ds, val_ds, test_ds, y_test_raw

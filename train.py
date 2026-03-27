"""
Script Principal d'Entraînement — Deep Learning Projet Fil Rouge
Auteur : ABTELX / ENSPD Deep Learning
Framework : TensorFlow / Keras

Usage :
    python train.py --mission cnn        # Entraîne la Mission 1 (CNN CIFAR-10)
    python train.py --mission lstm       # Entraîne la Mission 2 (LSTM Météo Jena)
    python train.py --mission both       # Entraîne les deux missions
"""

import os
import argparse
import numpy as np
import tensorflow as tf

# ── Imports locaux ────────────────────────────────────────────────────────────
from models.cnn_model  import CustomCNN
from models.rnn_model  import CustomLSTM
from utils.data_loader import load_cifar10, load_jena_climate, build_lstm_dataset
from utils.visualize   import (plot_training_history,
                                plot_confusion_matrix,
                                plot_lstm_predictions)

# ── Reproductibilité ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Dossiers de sortie ────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION 1 — CNN CIFAR-10
# ═══════════════════════════════════════════════════════════════════════════════

def train_cnn(epochs: int = 50, batch_size: int = 64):
    """
    Compile, entraîne et évalue le modèle CustomCNN sur CIFAR-10.

    Args:
        epochs     : Nombre maximum d'époques (EarlyStopping arrêtera avant si nécessaire).
        batch_size : Taille des mini-lots.
    """
    print("\n" + "="*60)
    print("  MISSION 1 — Classification d'Images CNN (CIFAR-10)")
    print("="*60)

    # ── 1. Données ────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_cifar10(batch_size=batch_size)

    # ── 2. Modèle ─────────────────────────────────────────────────────────────
    model = CustomCNN(num_classes=10, dropout_rate=0.5)

    # Construction du graphe pour model.summary()
    summary_model = model.build_graph(input_shape=(32, 32, 3))
    print("\n── Résumé de l'Architecture CNN ──")
    summary_model.summary()

    # ── 3. Compilation ────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        # Arrêt anticipé : surveille val_loss avec patience de 7 époques
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduction du LR si val_loss stagne
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Sauvegarde du meilleur modèle
        tf.keras.callbacks.ModelCheckpoint(
            filepath="saved_models/cnn_best.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # ── 5. Entraînement ───────────────────────────────────────────────────────
    print("\n── Démarrage de l'Entraînement CNN ──")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── 6. Évaluation sur le jeu de test ─────────────────────────────────────
    print("\n── Évaluation sur le Jeu de Test ──")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    if test_acc >= 0.70:
        print("✅ Objectif atteint : accuracy >= 70%")
    else:
        print("⚠️  Objectif non atteint. Essayez d'augmenter epochs ou d'ajuster le LR.")

    # ── 7. Visualisations ────────────────────────────────────────────────────
    # Courbes d'entraînement
    plot_training_history(history, save_path="results/cnn_training_history.png")

    # Matrice de confusion
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model(x_batch, training=False)
        y_pred.extend(np.argmax(preds.numpy(), axis=1))
        y_true.extend(y_batch.numpy())
    plot_confusion_matrix(np.array(y_true), np.array(y_pred),
                          save_path="results/cnn_confusion_matrix.png")

    # ── 8. Sauvegarde finale ─────────────────────────────────────────────────
    model.save("saved_models/cnn_final.keras")
    print("\n[Sauvegarde] Modèle CNN sauvegardé : saved_models/cnn_final.keras")

    return history, test_acc


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION 2 — LSTM MÉTÉO JENA
# ═══════════════════════════════════════════════════════════════════════════════

def train_lstm(epochs: int = 30, batch_size: int = 64,
               sequence_length: int = 24):
    """
    Compile, entraîne et évalue le modèle CustomLSTM sur le dataset Météo Jena.

    Args:
        epochs          : Nombre maximum d'époques.
        batch_size      : Taille des mini-lots.
        sequence_length : Fenêtre temporelle en entrée (heures de lookback).
    """
    print("\n" + "="*60)
    print("  MISSION 2 — Prédiction Météorologique LSTM (Jena Climate)")
    print("="*60)

    # ── 1. Données ────────────────────────────────────────────────────────────
    data_scaled, scaler, _ = load_jena_climate(data_dir="data/")
    train_ds, val_ds, test_ds, y_test_raw = build_lstm_dataset(
        data_scaled,
        sequence_length=sequence_length,
        batch_size=batch_size
    )

    # ── 2. Modèle ─────────────────────────────────────────────────────────────
    model = CustomLSTM(dropout_rate=0.2)

    summary_model = model.build_graph(sequence_length=sequence_length, num_features=1)
    print("\n── Résumé de l'Architecture LSTM ──")
    summary_model.summary()

    # ── 3. Compilation — MSE comme perte ET métrique ──────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mean_squared_error']
    )

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="saved_models/lstm_best.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # ── 5. Entraînement ───────────────────────────────────────────────────────
    print("\n── Démarrage de l'Entraînement LSTM ──")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── 6. Prédictions sur le jeu de test ────────────────────────────────────
    print("\n── Génération des Prédictions sur le Jeu de Test ──")
    y_pred_scaled = model.predict(test_ds, verbose=0)

    # Dé-normalisation via inverse_transform pour retrouver les vraies valeurs (°C)
    y_pred_celsius = scaler.inverse_transform(y_pred_scaled)
    y_true_celsius = scaler.inverse_transform(y_test_raw)

    # MSE en vraies valeurs
    mse = np.mean((y_true_celsius - y_pred_celsius) ** 2)
    rmse = np.sqrt(mse)
    print(f"Test MSE  (°C²) : {mse:.4f}")
    print(f"Test RMSE (°C)  : {rmse:.4f}")

    # ── 7. Visualisations ────────────────────────────────────────────────────
    plot_training_history(history, save_path="results/lstm_training_history.png")
    plot_lstm_predictions(y_true_celsius, y_pred_celsius, n_points=500,
                          save_path="results/lstm_predictions.png")

    # ── 8. Sauvegarde finale ─────────────────────────────────────────────────
    model.save("saved_models/lstm_final.keras")
    print("\n[Sauvegarde] Modèle LSTM sauvegardé : saved_models/lstm_final.keras")

    return history, rmse


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraînement Deep Learning — ENSPD Projet Fil Rouge"
    )
    parser.add_argument(
        "--mission",
        type=str,
        choices=["cnn", "lstm", "both"],
        default="both",
        help="Mission à entraîner : 'cnn', 'lstm', ou 'both' (défaut: both)"
    )
    parser.add_argument("--epochs",   type=int, default=50,  help="Nombre max d'époques")
    parser.add_argument("--batch",    type=int, default=64,  help="Taille des mini-lots")
    parser.add_argument("--seq_len",  type=int, default=24,  help="Fenêtre temporelle LSTM (heures)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n🚀 TensorFlow {tf.__version__} — GPUs détectés : {tf.config.list_physical_devices('GPU')}")

    if args.mission in ("cnn", "both"):
        train_cnn(epochs=args.epochs, batch_size=args.batch)

    if args.mission in ("lstm", "both"):
        train_lstm(epochs=args.epochs, batch_size=args.batch,
                   sequence_length=args.seq_len)

    print("\n✅ Entraînement terminé. Résultats dans : results/")

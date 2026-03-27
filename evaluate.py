"""
Script d'Évaluation — Chargement et test de modèles sauvegardés.
Auteur : ABTELX / ENSPD Deep Learning

Usage :
    python evaluate.py --mission cnn
    python evaluate.py --mission lstm
"""

import argparse
import numpy as np
import tensorflow as tf

from utils.data_loader import load_cifar10, load_jena_climate, build_lstm_dataset
from utils.visualize   import (plot_confusion_matrix, plot_lstm_predictions)


def evaluate_cnn(model_path: str = "saved_models/cnn_best.keras",
                 batch_size: int = 64):
    """
    Charge un modèle CNN sauvegardé et évalue ses performances sur CIFAR-10.

    Args:
        model_path : Chemin vers le fichier .keras ou .h5 du modèle.
        batch_size : Taille des mini-lots pour l'inférence.
    """
    print("\n" + "="*55)
    print("  ÉVALUATION CNN — CIFAR-10")
    print("="*55)

    # ── Chargement du modèle ─────────────────────────────────────────────────
    print(f"[Chargement] {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # ── Données de test ──────────────────────────────────────────────────────
    _, _, test_ds = load_cifar10(batch_size=batch_size)

    # ── Évaluation ───────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # ── Matrice de confusion ─────────────────────────────────────────────────
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model(x_batch, training=False)
        y_pred.extend(np.argmax(preds.numpy(), axis=1))
        y_true.extend(y_batch.numpy())

    plot_confusion_matrix(np.array(y_true), np.array(y_pred),
                          save_path="results/eval_cnn_confusion_matrix.png")
    print("✅ Évaluation CNN terminée.")


def evaluate_lstm(model_path: str = "saved_models/lstm_best.keras",
                  batch_size: int = 64, sequence_length: int = 24):
    """
    Charge un modèle LSTM sauvegardé et génère les prédictions sur Jena.

    Args:
        model_path      : Chemin vers le fichier .keras ou .h5 du modèle.
        batch_size      : Taille des mini-lots pour l'inférence.
        sequence_length : Fenêtre temporelle utilisée lors de l'entraînement.
    """
    print("\n" + "="*55)
    print("  ÉVALUATION LSTM — Météo Jena")
    print("="*55)

    # ── Chargement du modèle ─────────────────────────────────────────────────
    print(f"[Chargement] {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # ── Données de test ──────────────────────────────────────────────────────
    data_scaled, scaler, _ = load_jena_climate(data_dir="data/")
    _, _, test_ds, y_test_raw = build_lstm_dataset(
        data_scaled,
        sequence_length=sequence_length,
        batch_size=batch_size
    )

    # ── Prédictions ──────────────────────────────────────────────────────────
    y_pred_scaled  = model.predict(test_ds, verbose=1)
    y_pred_celsius = scaler.inverse_transform(y_pred_scaled)
    y_true_celsius = scaler.inverse_transform(y_test_raw)

    # ── Métriques ────────────────────────────────────────────────────────────
    mse  = np.mean((y_true_celsius - y_pred_celsius) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y_true_celsius - y_pred_celsius))
    print(f"\nTest MSE  (°C²) : {mse:.4f}")
    print(f"Test RMSE (°C)  : {rmse:.4f}")
    print(f"Test MAE  (°C)  : {mae:.4f}")

    # ── Graphique réel vs prédit ──────────────────────────────────────────────
    plot_lstm_predictions(y_true_celsius, y_pred_celsius, n_points=500,
                          save_path="results/eval_lstm_predictions.png")
    print("✅ Évaluation LSTM terminée.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évaluation de modèles Deep Learning sauvegardés"
    )
    parser.add_argument(
        "--mission",
        type=str,
        choices=["cnn", "lstm"],
        required=True,
        help="Modèle à évaluer : 'cnn' ou 'lstm'"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Chemin personnalisé vers le fichier modèle (.keras / .h5)"
    )
    parser.add_argument("--batch",   type=int, default=64, help="Taille des mini-lots")
    parser.add_argument("--seq_len", type=int, default=24, help="Fenêtre temporelle LSTM")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mission == "cnn":
        path = args.model_path or "saved_models/cnn_best.keras"
        evaluate_cnn(model_path=path, batch_size=args.batch)

    elif args.mission == "lstm":
        path = args.model_path or "saved_models/lstm_best.keras"
        evaluate_lstm(model_path=path, batch_size=args.batch,
                      sequence_length=args.seq_len)

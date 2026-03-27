"""
Fonctions de visualisation pour les deux missions.
Auteur : ABTELX / ENSPD Deep Learning

Contenu :
    - plot_training_history()   : Courbes Loss / Accuracy (History de model.fit)
    - plot_confusion_matrix()   : Matrice de confusion annotée (CNN)
    - plot_lstm_predictions()   : Courbe réelle vs prédite (LSTM)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # Backend sans affichage GUI (compatible serveur/Colab)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Noms des classes CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ═══════════════════════════════════════════════════════════════════════════════
# COURBES D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path: str = "results/cnn_training_history.png"):
    """
    Génère et sauvegarde les courbes Train Loss vs Validation Loss
    et Train Accuracy vs Validation Accuracy issues de l'objet History.

    Args:
        history   : Objet retourné par model.fit().
        save_path : Chemin de sauvegarde de la figure.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'Entraînement", fontsize=14, fontweight='bold')

    # ── Loss ────────────────────────────────────────────────────────────────
    axes[0].plot(history.history['loss'],     label='Train Loss',      color='#2196F3', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#F44336',
                 linewidth=2, linestyle='--')
    axes[0].set_title("Fonction de Perte")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy (si présente dans l'historique) ─────────────────────────
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'],     label='Train Accuracy',      color='#4CAF50', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF9800',
                     linewidth=2, linestyle='--')
        axes[1].set_title("Précision (Accuracy)")
        axes[1].set_xlabel("Époque")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Pour le LSTM : on affiche MSE au lieu d'accuracy
        axes[1].plot(history.history.get('mean_squared_error', history.history['loss']),
                     label='Train MSE', color='#4CAF50', linewidth=2)
        axes[1].plot(history.history.get('val_mean_squared_error', history.history['val_loss']),
                     label='Val MSE', color='#FF9800', linewidth=2, linestyle='--')
        axes[1].set_title("Erreur Quadratique Moyenne (MSE)")
        axes[1].set_xlabel("Époque")
        axes[1].set_ylabel("MSE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Courbes sauvegardées : {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MATRICE DE CONFUSION (CNN)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str = "results/cnn_confusion_matrix.png"):
    """
    Génère et sauvegarde la matrice de confusion annotée pour CIFAR-10.

    Args:
        y_true    : Labels réels (array 1D d'entiers).
        y_pred    : Labels prédits (array 1D d'entiers).
        save_path : Chemin de sauvegarde de la figure.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)
    ax.set_title("Matrice de Confusion — CIFAR-10", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Matrice de confusion sauvegardée : {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# COURBE RÉELLE VS PRÉDITE (LSTM)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lstm_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                          n_points: int = 500,
                          save_path: str = "results/lstm_predictions.png"):
    """
    Superpose la courbe réelle et la courbe prédite sur les données de test.
    Les valeurs doivent être dé-normalisées (inverse_transform) avant l'appel.

    Args:
        y_true    : Vraies valeurs de température (dé-normalisées).
        y_pred    : Valeurs prédites par le LSTM (dé-normalisées).
        n_points  : Nombre de points à afficher (pour lisibilité).
        save_path : Chemin de sauvegarde de la figure.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_true = np.array(y_true).flatten()[:n_points]
    y_pred = np.array(y_pred).flatten()[:n_points]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="Valeurs Réelles",  color='#2196F3', linewidth=1.5, alpha=0.9)
    ax.plot(y_pred, label="Prédictions LSTM", color='#F44336', linewidth=1.5,
            linestyle='--', alpha=0.9)
    ax.set_title("Prédiction de Température — Données Météo Jena (T+1)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Pas de temps (heures)")
    ax.set_ylabel("Température (°C)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Courbe LSTM sauvegardée : {save_path}")

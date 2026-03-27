# Projet Fil Rouge — Deep Learning (ENSPD)

**Framework :** TensorFlow / Keras  
**Auteur :** ABTELX / ENSPD  
**Évaluation :** Dr. Noulapeu N. A.

---

## 📁 Structure du Projet

```
projet_deep_learning/
├── data/                          ← Jeux de données bruts (auto-téléchargés)
├── models/
│   ├── cnn_model.py               ← Classe CustomCNN (API Subclassing)
│   └── rnn_model.py               ← Classe CustomLSTM (API Subclassing)
├── utils/
│   ├── data_loader.py             ← Pipelines tf.data.Dataset
│   └── visualize.py               ← Fonctions de visualisation
├── train.py                       ← Script principal d'entraînement
├── evaluate.py                    ← Script d'évaluation des modèles sauvegardés
├── requirements.txt               ← Dépendances Python
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/projet-deep-learning-enspd.git
cd projet-deep-learning-enspd

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

> 💡 **Google Colab** : Tous les packages sont pré-installés. Uploadez le projet et lancez directement.

---

## 🚀 Lancer l'Entraînement

```bash
# Mission 1 + Mission 2 (défaut)
python train.py --mission both

# Mission 1 uniquement (CNN CIFAR-10)
python train.py --mission cnn --epochs 50 --batch 64

# Mission 2 uniquement (LSTM Météo Jena)
python train.py --mission lstm --epochs 30 --batch 64 --seq_len 24
```

Les données CIFAR-10 et Jena Climate sont **téléchargées automatiquement** au premier lancement.

---

## 📊 Évaluation d'un Modèle Sauvegardé

```bash
# Évaluer le meilleur modèle CNN
python evaluate.py --mission cnn

# Évaluer le meilleur modèle LSTM
python evaluate.py --mission lstm

# Spécifier un fichier modèle personnalisé
python evaluate.py --mission cnn --model_path saved_models/cnn_final.keras
```

---

## 📈 Résultats Générés

Après entraînement, les fichiers suivants sont créés dans `results/` :

| Fichier | Contenu |
|---------|---------|
| `cnn_training_history.png`  | Courbes Train/Val Loss + Accuracy (CNN) |
| `cnn_confusion_matrix.png`  | Matrice de confusion CIFAR-10 (10 classes) |
| `lstm_training_history.png` | Courbes Train/Val MSE (LSTM) |
| `lstm_predictions.png`      | Courbe réelle vs prédite — Température Jena |

---

## 🧠 Architectures

### Mission 1 — CustomCNN

| Couche | Paramètres |
|--------|-----------|
| Data Augmentation | RandomFlip + RandomRotation + RandomZoom |
| Conv2D × 3 | 32 → 64 → 128 filtres, kernel 3×3, padding='same' |
| BatchNormalization × 3 | Après chaque Conv2D |
| MaxPooling2D × 3 | Pool 2×2 |
| Flatten | — |
| Dense(256) + Dropout(0.5) | Régularisation |
| Dense(128) | — |
| Dense(10, softmax) | Sortie — 10 classes |

**Objectif :** Accuracy ≥ 70% sur le jeu de test CIFAR-10.

### Mission 2 — CustomLSTM

| Couche | Paramètres |
|--------|-----------|
| LSTM(128) | return_sequences=True |
| Dropout(0.2) | — |
| LSTM(64) | return_sequences=False |
| Dropout(0.2) | — |
| Dense(32, relu) | — |
| Dense(1) | Prédiction T+1 (température °C) |

**Perte :** MSE | **Données :** Météo Jena Climate 2009-2016 | **Fenêtre :** 24h

---

## 📦 Dépendances

```
tensorflow >= 2.13.0
numpy      >= 1.24.0
matplotlib >= 3.7.0
scikit-learn >= 1.3.0
pandas     >= 2.0.0
```

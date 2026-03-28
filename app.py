"""
Interface de Présentation — Projet Fil Rouge Deep Learning
ENSPD | Dr. Noulapeu N. A.
Framework : Streamlit + TensorFlow / Keras

Lancement : streamlit run app.py
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

# ── Configuration Streamlit ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Learning — ENSPD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS GLOBAL — Design sombre professionnel
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Variables ── */
:root {
    --bg-primary:   #0A0E1A;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --accent:       #00D4FF;
    --accent2:      #7C3AED;
    --accent3:      #10B981;
    --warn:         #F59E0B;
    --danger:       #EF4444;
    --text-primary: #F1F5F9;
    --text-muted:   #94A3B8;
    --border:       rgba(0,212,255,0.15);
    --glow:         0 0 20px rgba(0,212,255,0.15);
}

/* ── Base ── */
.stApp { background: var(--bg-primary); font-family: 'DM Sans', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #111827 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox label { color: var(--text-muted) !important; }

/* ── Titres ── */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: var(--accent) !important; font-size: 1.6rem !important; letter-spacing: -0.5px; }
h2 { color: var(--text-primary) !important; font-size: 1.2rem !important; }
h3 { color: var(--accent) !important; font-size: 1rem !important; }

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--glow);
    transition: border-color 0.3s ease;
}
.card:hover { border-color: rgba(0,212,255,0.4); }

.card-accent {
    background: linear-gradient(135deg, #0D1B2E 0%, #111827 100%);
    border-left: 3px solid var(--accent);
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Métriques ── */
.metric-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-blue  { background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.3); }
.badge-green { background: rgba(16,185,129,0.12); color: var(--accent3); border: 1px solid rgba(16,185,129,0.3); }
.badge-purple{ background: rgba(124,58,237,0.12); color: #A78BFA; border: 1px solid rgba(124,58,237,0.3); }
.badge-warn  { background: rgba(245,158,11,0.12); color: var(--warn); border: 1px solid rgba(245,158,11,0.3); }

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0D1B2E 0%, #1a0e2e 50%, #0A1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;  left: -20%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,212,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
}

/* ── Prediction bar ── */
.pred-bar-container { margin: 0.4rem 0; }
.pred-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 3px;
}
.pred-bar-outer {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.pred-bar-inner {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.5s ease;
}
.pred-bar-inner.top {
    background: linear-gradient(90deg, var(--accent), var(--accent3));
    box-shadow: 0 0 6px rgba(0,212,255,0.5);
}

/* ── Info box ── */
.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px dashed rgba(0,212,255,0.25);
    border-radius: 8px;
    padding: 1rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    text-align: center;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,212,255,0.3) !important;
    border-radius: 10px !important;
    background: rgba(0,212,255,0.03) !important;
}

div[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem !important;
}
div[data-testid="metric-container"] label { color: var(--text-muted) !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color: var(--accent) !important; }

/* Plots background */
.stPlotlyChart, .element-container img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
CIFAR10_CLASSES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird", "🐱 Cat", "🦌 Deer",
    "🐶 Dog", "🐸 Frog", "🐴 Horse", "🚢 Ship", "🚛 Truck"
]
CIFAR10_CLASSES_CLEAN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

plt.rcParams.update({
    'figure.facecolor': '#111827',
    'axes.facecolor':   '#111827',
    'axes.edgecolor':   '#1E293B',
    'axes.labelcolor':  '#94A3B8',
    'xtick.color':      '#94A3B8',
    'ytick.color':      '#94A3B8',
    'text.color':       '#F1F5F9',
    'grid.color':       '#1E293B',
    'grid.alpha':       1.0,
})


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Chargement du modèle CNN...")
def load_cnn_model():
    import tensorflow as tf
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.cnn_model import CustomCNN
    path = "saved_models/cnn_best.keras"
    if not os.path.exists(path):
        path = "saved_models/cnn_final.keras"
    if os.path.exists(path):
        return tf.keras.models.load_model(
            path, custom_objects={"CustomCNN": CustomCNN}, compile=False)
    return None

@st.cache_resource(show_spinner="⏳ Chargement du modèle LSTM...")
def load_lstm_model():
    import tensorflow as tf
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.rnn_model import CustomLSTM
    path = "saved_models/lstm_best.keras"
    if not os.path.exists(path):
        path = "saved_models/lstm_final.keras"
    if os.path.exists(path):
        return tf.keras.models.load_model(
            path, custom_objects={"CustomLSTM": CustomLSTM}, compile=False)
    return None

def fig_to_image(fig):
    """Convertit une figure Matplotlib en bytes PNG pour Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def make_confidence_bars(probs, classes, top_n=5):
    """Génère le HTML des barres de confiance."""
    top_idx = np.argsort(probs)[::-1][:top_n]
    html = ""
    for rank, idx in enumerate(top_idx):
        pct = probs[idx] * 100
        is_top = rank == 0
        bar_class = "top" if is_top else ""
        label_color = "#00D4FF" if is_top else "#94A3B8"
        html += f"""
        <div class="pred-bar-container">
            <div class="pred-bar-label">
                <span style="color:{label_color};font-weight:{'700' if is_top else '400'}">{classes[idx]}</span>
                <span style="color:{label_color};font-family:'Space Mono',monospace">{pct:.1f}%</span>
            </div>
            <div class="pred-bar-outer">
                <div class="pred-bar-inner {bar_class}" style="width:{pct:.1f}%"></div>
            </div>
        </div>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem">
        <div style="font-size:2.5rem">🧠</div>
        <div style="font-family:'Space Mono',monospace;font-size:1rem;
                    color:#00D4FF;font-weight:700;margin-top:0.4rem">
            DEEP LEARNING
        </div>
        <div style="font-size:0.72rem;color:#64748B;margin-top:0.2rem;
                    text-transform:uppercase;letter-spacing:1.5px">
            ENSPD · Projet Fil Rouge
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(0,212,255,0.15);margin:0.8rem 0">
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        options=[
            "🏠  Accueil",
            "🖼️  Mission 1 — CNN",
            "📈  Mission 2 — LSTM",
            "📊  Résultats & Courbes",
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(0,212,255,0.1);margin:1rem 0 0.8rem">
    <div style="font-size:0.7rem;color:#475569;text-align:center;line-height:1.8">
        <div><span style="color:#00D4FF">▸</span> TensorFlow / Keras</div>
        <div><span style="color:#7C3AED">▸</span> API Subclassing</div>
        <div><span style="color:#10B981">▸</span> CIFAR-10 · Jena Climate</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if "Accueil" in page:

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Projet Fil Rouge<br>Deep Learning</div>
        <div class="hero-subtitle">
            ENSPD Douala · Évaluation Pratique · Dr. Noulapeu N. A.<br>
            Framework : TensorFlow / Keras · API Subclassing
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Métriques projet
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">2</div>
            <div class="metric-label">Missions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">70%+</div>
            <div class="metric-label">Objectif CNN</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">60K</div>
            <div class="metric-label">Images CIFAR-10</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">420K</div>
            <div class="metric-label">Mesures Jena</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card-accent">
            <div class="section-header">
                <div class="section-dot"></div>
                <h2 style="margin:0">Mission 1 — Vision par Ordinateur</h2>
            </div>
            <p style="color:#94A3B8;font-size:0.88rem;line-height:1.7;margin:0">
                Classification de 10 catégories d'images sur le benchmark
                <strong style="color:#F1F5F9">CIFAR-10</strong>.
                Architecture CNN custom avec 3 blocs convolutifs,
                BatchNormalization, MaxPooling, Dropout et Data Augmentation
                intégrée.
            </p>
            <div style="margin-top:1rem;display:flex;gap:0.5rem;flex-wrap:wrap">
                <span class="badge badge-blue">Conv2D × 3</span>
                <span class="badge badge-green">BatchNorm</span>
                <span class="badge badge-purple">Dropout 50%</span>
                <span class="badge badge-warn">EarlyStopping</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card-accent">
            <div class="section-header">
                <div class="section-dot" style="background:#7C3AED;box-shadow:0 0 8px #7C3AED"></div>
                <h2 style="margin:0">Mission 2 — Séries Temporelles</h2>
            </div>
            <p style="color:#94A3B8;font-size:0.88rem;line-height:1.7;margin:0">
                Prédiction de la <strong style="color:#F1F5F9">température T+1</strong>
                sur le dataset météo de Jena (2009–2016).
                Architecture LSTM avec fenêtres glissantes de 24h,
                normalisation MinMaxScaler et visualisation réel vs prédit.
            </p>
            <div style="margin-top:1rem;display:flex;gap:0.5rem;flex-wrap:wrap">
                <span class="badge badge-purple">LSTM × 2</span>
                <span class="badge badge-blue">Sliding Window 24h</span>
                <span class="badge badge-green">MinMaxScaler</span>
                <span class="badge badge-warn">MSE Loss</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Architecture logicielle
    st.markdown("""
    <div class="card">
        <div class="section-header">
            <div class="section-dot" style="background:#10B981;box-shadow:0 0 8px #10B981"></div>
            <h2 style="margin:0">Architecture Logicielle — Skeleton</h2>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-top:0.5rem">
            <div style="background:#0A0E1A;border:1px solid rgba(0,212,255,0.1);border-radius:8px;padding:0.8rem">
                <div style="font-family:'Space Mono',monospace;color:#00D4FF;font-size:0.75rem">models/</div>
                <div style="color:#64748B;font-size:0.72rem;margin-top:0.3rem">cnn_model.py<br>rnn_model.py</div>
            </div>
            <div style="background:#0A0E1A;border:1px solid rgba(124,58,237,0.2);border-radius:8px;padding:0.8rem">
                <div style="font-family:'Space Mono',monospace;color:#A78BFA;font-size:0.75rem">utils/</div>
                <div style="color:#64748B;font-size:0.72rem;margin-top:0.3rem">data_loader.py<br>visualize.py</div>
            </div>
            <div style="background:#0A0E1A;border:1px solid rgba(16,185,129,0.2);border-radius:8px;padding:0.8rem">
                <div style="font-family:'Space Mono',monospace;color:#10B981;font-size:0.75rem">scripts/</div>
                <div style="color:#64748B;font-size:0.72rem;margin-top:0.3rem">train.py<br>evaluate.py</div>
            </div>
            <div style="background:#0A0E1A;border:1px solid rgba(245,158,11,0.2);border-radius:8px;padding:0.8rem">
                <div style="font-family:'Space Mono',monospace;color:#F59E0B;font-size:0.75rem">config/</div>
                <div style="color:#64748B;font-size:0.72rem;margin-top:0.3rem">requirements.txt<br>README.md</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : MISSION 1 — CNN
# ══════════════════════════════════════════════════════════════════════════════
elif "CNN" in page:

    st.markdown("# 🖼️ Mission 1 — Classification d'Images CNN")

    model = load_cnn_model()

    if model is None:
        st.markdown("""
        <div class="info-box">
            ⚠️ Aucun modèle CNN trouvé dans <code>saved_models/</code>.<br>
            Lance d'abord : <code>python train.py --mission cnn</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="background:rgba(16,185,129,0.05);border-color:rgba(16,185,129,0.3)">
            ✅ &nbsp;<strong style="color:#10B981">Modèle CNN chargé avec succès</strong>
        </div>
        """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div class="section-header">
            <div class="section-dot"></div>
            <h2 style="margin:0">Uploader une Image</h2>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Glisse une image (JPG, PNG) — idéalement une des 10 catégories CIFAR-10",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown("""
        <div style="margin-top:0.8rem">
            <div style="font-size:0.75rem;color:#64748B;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:0.6rem">Catégories CIFAR-10</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.4rem">
        """ + "".join([f'<span class="badge badge-blue">{c}</span>' for c in CIFAR10_CLASSES])
        + "</div></div>", unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Image uploadée", use_container_width=True)

    with col_result:
        st.markdown("""
        <div class="section-header">
            <div class="section-dot" style="background:#7C3AED;box-shadow:0 0 8px #7C3AED"></div>
            <h2 style="margin:0">Prédiction du Modèle</h2>
        </div>
        """, unsafe_allow_html=True)

        if uploaded and model is not None:
            import tensorflow as tf

            # Preprocessing
            img_resized = img.resize((32, 32))
            img_array  = np.array(img_resized).astype("float32") / 255.0
            img_batch  = np.expand_dims(img_array, axis=0)

            # Prédiction
            with st.spinner("Inférence en cours..."):
                probs = model(img_batch, training=False).numpy()[0]

            top_idx   = np.argmax(probs)
            top_class = CIFAR10_CLASSES[top_idx]
            top_conf  = probs[top_idx] * 100

            # Résultat principal
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(124,58,237,0.08));
                        border:1px solid rgba(0,212,255,0.3);border-radius:12px;
                        padding:1.5rem;text-align:center;margin-bottom:1.2rem">
                <div style="font-size:2.5rem;margin-bottom:0.3rem">{top_class.split()[0]}</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.5rem;
                            color:#00D4FF;font-weight:700">{' '.join(top_class.split()[1:]).upper()}</div>
                <div style="font-size:2rem;font-weight:700;color:#10B981;
                            font-family:'Space Mono',monospace;margin-top:0.3rem">
                    {top_conf:.1f}%
                </div>
                <div style="font-size:0.75rem;color:#64748B;text-transform:uppercase;
                            letter-spacing:1px">confiance</div>
            </div>
            """, unsafe_allow_html=True)

            # Barres de confiance Top 5
            st.markdown("<div style='font-size:0.8rem;color:#64748B;text-transform:uppercase;"
                        "letter-spacing:1px;margin-bottom:0.6rem'>Top 5 Probabilités</div>",
                        unsafe_allow_html=True)
            st.markdown(make_confidence_bars(probs, CIFAR10_CLASSES, top_n=5),
                        unsafe_allow_html=True)

        elif model is None:
            st.markdown("""
            <div class="info-box" style="margin-top:3rem">
                🔴 Modèle non disponible.<br>Lance l'entraînement d'abord.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="margin-top:3rem">
                ⬅️ Upload une image pour voir la prédiction du CNN
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : MISSION 2 — LSTM
# ══════════════════════════════════════════════════════════════════════════════
elif "LSTM" in page:

    st.markdown("# 📈 Mission 2 — Prédiction Météorologique LSTM")

    model_lstm = load_lstm_model()

    if model_lstm is None:
        st.markdown("""
        <div class="info-box">
            ⚠️ Aucun modèle LSTM trouvé dans <code>saved_models/</code>.<br>
            Lance d'abord : <code>python train.py --mission lstm</code>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="background:rgba(16,185,129,0.05);border-color:rgba(16,185,129,0.3)">
            ✅ &nbsp;<strong style="color:#10B981">Modèle LSTM chargé avec succès</strong>
        </div>
        """, unsafe_allow_html=True)

        # Contrôles
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            n_points = st.slider("Nombre de points affichés", 100, 1000, 500, step=50)
        with col_ctrl2:
            seq_len = st.selectbox("Fenêtre temporelle (h)", [12, 24, 48], index=1)
        with col_ctrl3:
            show_error = st.toggle("Afficher bande d'erreur", value=True)

        # Chargement données
        @st.cache_data(show_spinner="📡 Chargement des données Jena...")
        def get_jena_predictions(seq_length, n_pts):
            try:
                import tensorflow as tf
                from sklearn.preprocessing import MinMaxScaler
                import pandas as pd
                csv_candidates = [
                    "data/jena_climate_2009_2016.csv",
                    os.path.join(os.path.expanduser("~"), ".keras", "datasets",
                                 "jena_climate_2009_2016.csv")
                ]
                csv_path = None
                for p in csv_candidates:
                    if os.path.exists(p):
                        csv_path = p
                        break

                if csv_path is None:
                    return None, None, None, None

                df = pd.read_csv(csv_path)
                df = df[5::6].reset_index(drop=True)
                temperature = df[["T (degC)"]].values.astype("float32")
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(temperature)

                val_end   = int(len(scaled) * 0.85)
                test_data = scaled[val_end:]

                ds = tf.keras.utils.timeseries_dataset_from_array(
                    data=test_data[:-1],
                    targets=test_data[seq_length:],
                    sequence_length=seq_length,
                    batch_size=64
                )
                _m = load_lstm_model()
                y_pred = _m.predict(ds, verbose=0)
                y_true = test_data[seq_length:]

                y_pred_c = scaler.inverse_transform(y_pred)
                y_true_c = scaler.inverse_transform(y_true)
                return y_true_c[:n_pts], y_pred_c[:n_pts], scaler, None
            except Exception as e:
                return None, None, None, str(e)

        y_true, y_pred, _, err = get_jena_predictions(seq_len, n_points)

        if err:
            st.error(f"Erreur : {err}")
        elif y_true is None:
            st.markdown("""
            <div class="info-box">
                📂 Dataset Jena non trouvé. Lance d'abord <code>python train.py --mission lstm</code>
                pour télécharger les données automatiquement.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Métriques
            mse  = float(np.mean((y_true - y_pred)**2))
            rmse = float(np.sqrt(mse))
            mae  = float(np.mean(np.abs(y_true - y_pred)))

            m1, m2, m3 = st.columns(3)
            m1.metric("MSE (°C²)", f"{mse:.4f}")
            m2.metric("RMSE (°C)", f"{rmse:.4f}")
            m3.metric("MAE (°C)",  f"{mae:.4f}")

            # Graphique
            fig, ax = plt.subplots(figsize=(13, 4.5))
            x = np.arange(len(y_true))
            ax.plot(x, y_true.flatten(), color="#00D4FF", lw=1.5,
                    label="Température Réelle", alpha=0.95)
            ax.plot(x, y_pred.flatten(), color="#F59E0B", lw=1.5,
                    linestyle="--", label="Prédiction LSTM", alpha=0.9)
            if show_error:
                err_band = np.abs(y_true.flatten() - y_pred.flatten())
                ax.fill_between(x,
                    y_pred.flatten() - err_band,
                    y_pred.flatten() + err_band,
                    color="#7C3AED", alpha=0.15, label="Bande d'erreur")
            ax.set_title(f"Prédiction Météorologique Jena — T+1 · {n_points} points",
                         fontsize=12, pad=12)
            ax.set_xlabel("Pas de temps (heures)")
            ax.set_ylabel("Température (°C)")
            ax.legend(fontsize=9, framealpha=0.2)
            ax.grid(True, alpha=0.15)
            fig.tight_layout()
            st.image(fig_to_image(fig), use_container_width=True)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : RÉSULTATS & COURBES
# ══════════════════════════════════════════════════════════════════════════════
elif "Résultats" in page:

    st.markdown("# 📊 Résultats & Courbes d'Entraînement")

    RESULTS_DIR = "results"

    def load_result_image(filename):
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            return Image.open(path)
        return None

    # ── CNN ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div>
        <h2 style="margin:0">Mission 1 — CNN CIFAR-10</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("<div style='font-size:0.8rem;color:#64748B;margin-bottom:0.5rem;"
                    "text-transform:uppercase;letter-spacing:1px'>Courbes d'entraînement</div>",
                    unsafe_allow_html=True)
        img_hist = load_result_image("cnn_training_history.png")
        if img_hist:
            st.image(img_hist, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box">
                📂 Fichier non trouvé : <code>results/cnn_training_history.png</code><br>
                Lance <code>python train.py --mission cnn</code> pour générer ce graphique.
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='font-size:0.8rem;color:#64748B;margin-bottom:0.5rem;"
                    "text-transform:uppercase;letter-spacing:1px'>Matrice de confusion</div>",
                    unsafe_allow_html=True)
        img_cm = load_result_image("cnn_confusion_matrix.png")
        if img_cm:
            st.image(img_cm, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box">
                📂 Fichier non trouvé : <code>results/cnn_confusion_matrix.png</code><br>
                Lance <code>python train.py --mission cnn</code> pour générer ce graphique.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LSTM ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <div class="section-dot" style="background:#7C3AED;box-shadow:0 0 8px #7C3AED"></div>
        <h2 style="margin:0">Mission 2 — LSTM Météo Jena</h2>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown("<div style='font-size:0.8rem;color:#64748B;margin-bottom:0.5rem;"
                    "text-transform:uppercase;letter-spacing:1px'>Courbes MSE</div>",
                    unsafe_allow_html=True)
        img_lstm_hist = load_result_image("lstm_training_history.png")
        if img_lstm_hist:
            st.image(img_lstm_hist, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box">
                📂 Fichier non trouvé : <code>results/lstm_training_history.png</code><br>
                Lance <code>python train.py --mission lstm</code> pour générer ce graphique.
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown("<div style='font-size:0.8rem;color:#64748B;margin-bottom:0.5rem;"
                    "text-transform:uppercase;letter-spacing:1px'>Réel vs Prédit</div>",
                    unsafe_allow_html=True)
        img_lstm_pred = load_result_image("lstm_predictions.png")
        if img_lstm_pred:
            st.image(img_lstm_pred, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box">
                📂 Fichier non trouvé : <code>results/lstm_predictions.png</code><br>
                Lance <code>python train.py --mission lstm</code> pour générer ce graphique.
            </div>
            """, unsafe_allow_html=True)

    # ── Résumé final ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="section-header">
            <div class="section-dot" style="background:#10B981;box-shadow:0 0 8px #10B981"></div>
            <h2 style="margin:0">Résumé des Performances</h2>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
            <div style="background:#0A0E1A;border:1px solid rgba(0,212,255,0.1);
                        border-radius:8px;padding:1rem">
                <div style="font-family:'Space Mono',monospace;color:#00D4FF;
                            font-size:0.8rem;margin-bottom:0.6rem">CNN · CIFAR-10</div>
                <div style="color:#94A3B8;font-size:0.85rem;line-height:1.8">
                    Objectif : Accuracy ≥ 70% · Optimiseur : Adam · Loss : SparseCCE<br>
                    Régularisation : Dropout(0.5) + BatchNorm + EarlyStopping(patience=7)
                </div>
            </div>
            <div style="background:#0A0E1A;border:1px solid rgba(124,58,237,0.15);
                        border-radius:8px;padding:1rem">
                <div style="font-family:'Space Mono',monospace;color:#A78BFA;
                            font-size:0.8rem;margin-bottom:0.6rem">LSTM · Jena Climate</div>
                <div style="color:#94A3B8;font-size:0.85rem;line-height:1.8">
                    Prédiction T+1 · Fenêtre : 24h · Optimiseur : Adam · Loss : MSE<br>
                    Régularisation : Dropout(0.2) + EarlyStopping(patience=5)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
import streamlit as st
from model_utils import (
    load_medical_cnn, predict_mri_image,
    load_bio_model, predict_bio,
    obtenir_recommandations_neuro, obtenir_recommandations_bio,
    generer_pdf_neuro, generer_pdf_bio
)
from PIL import Image

st.set_page_config(page_title="VET-AI | Diagnostic Platform", page_icon="🧬", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace !important;
        background-color: #020817 !important;
    }

    .stApp { background: #020817; }

    /* Header hero */
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 24px;
        padding: 40px 30px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(16,185,129,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #10B981, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        color: #818CF8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 16px;
        letter-spacing: 1px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 6px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748B;
        border-radius: 12px;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        padding: 10px 24px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: white !important;
    }

    /* Section headers */
    .section-header {
        font-family: 'Syne', sans-serif;
        color: #E2E8F0;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 3px solid #6366F1;
        padding-left: 12px;
        margin: 24px 0 16px 0;
        letter-spacing: 0.5px;
    }

    /* Result cards */
    .card-sain {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.1) 100%);
        border: 1px solid rgba(16,185,129,0.4);
        padding: 24px;
        border-radius: 20px;
        color: white;
        margin: 16px 0;
        position: relative;
        overflow: hidden;
    }
    .card-sain::before {
        content: '✓';
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 4rem;
        color: rgba(16,185,129,0.15);
        font-weight: 800;
    }
    .card-malade {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(220,38,38,0.1) 100%);
        border: 1px solid rgba(239,68,68,0.4);
        padding: 24px;
        border-radius: 20px;
        color: white;
        margin: 16px 0;
        position: relative;
        overflow: hidden;
    }
    .card-malade::before {
        content: '!';
        position: absolute;
        right: 24px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 4rem;
        color: rgba(239,68,68,0.15);
        font-weight: 800;
    }
    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .card-sub {
        font-size: 0.85rem;
        opacity: 0.8;
        margin: 0;
        font-family: 'DM Mono', monospace;
    }

    /* Confidence bar */
    .conf-bar-container {
        background: #1E293B;
        border-radius: 8px;
        height: 8px;
        margin: 12px 0 4px 0;
        overflow: hidden;
    }
    .conf-bar-fill-green {
        height: 100%;
        background: linear-gradient(90deg, #10B981, #34D399);
        border-radius: 8px;
        transition: width 1s ease;
    }
    .conf-bar-fill-red {
        height: 100%;
        background: linear-gradient(90deg, #EF4444, #F87171);
        border-radius: 8px;
    }

    /* Recommandation items */
    .rx-item {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 3px solid #6366F1;
        padding: 14px 18px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 10px;
        color: #CBD5E1;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Input styling */
    .stNumberInput input, .stSlider {
        background: #0F172A !important;
        border-color: #334155 !important;
        color: #E2E8F0 !important;
    }
    label { color: #94A3B8 !important; font-size: 0.85rem !important; }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #6366F1;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Download button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 20px !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        margin-top: 16px !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #6366F1 !important; }

    /* File uploader */
    .stFileUploader {
        background: #0F172A !important;
        border: 2px dashed #334155 !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #334155;
        font-size: 0.75rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1E293B;
        letter-spacing: 1px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-subtitle">École Nationale Supérieure Vétérinaire • IA Médicale</p>
    <h1 class="hero-title">🧬 VET-AI Platform</h1>
    <p style="color:#64748B; font-size:0.9rem; margin-top:10px; font-family:'DM Mono',monospace;">
        Système de Diagnostic Assisté par Intelligence Artificielle
    </p>
    <span class="hero-badge">Random Forest Classifier • v2.0 • Accuracy 96.9%</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧠  Analyse IRM Cérébrale", "🔬  Diagnostic Biologique"])

# ════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE IRM
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">📁 Téléversement du Cliché IRM</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; font-size:0.85rem; margin-bottom:16px;">Importez une image IRM cérébrale au format JPG ou PNG pour analyse automatique.</p>', unsafe_allow_html=True)

    model_irm = load_medical_cnn()
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="irm_uploader",
                                      label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="Cliché IRM — Prêt pour l'analyse", use_container_width=True)

        with st.spinner("⚡ Analyse du parenchyme cérébral en cours..."):
            prediction, confidence = predict_mri_image(model_irm, image)

        st.markdown('<p class="section-header">🎯 Résultat Diagnostique</p>', unsafe_allow_html=True)

        if prediction == 0:
            st.markdown(f"""
            <div class="card-sain">
                <p class="card-title">✅ EXAMEN NORMAL — SAIN</p>
                <p class="card-sub">Aucune lésion expansive détectée par le réseau neuronal</p>
                <div class="conf-bar-container">
                    <div class="conf-bar-fill-green" style="width:{confidence:.1f}%"></div>
                </div>
                <p style="color:#10B981; font-size:0.8rem; margin:0; font-family:'DM Mono',monospace;">
                    Confiance algorithmique : {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-malade">
                <p class="card-title">🚨 ANOMALIE — TUMEUR DÉTECTÉE</p>
                <p class="card-sub">Suspicion de processus tumoral — Consultation urgente requise</p>
                <div class="conf-bar-container">
                    <div class="conf-bar-fill-red" style="width:{confidence:.1f}%"></div>
                </div>
                <p style="color:#EF4444; font-size:0.8rem; margin:0; font-family:'DM Mono',monospace;">
                    Confiance algorithmique : {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">💊 Orientations Cliniques</p>', unsafe_allow_html=True)
        recommandations = obtenir_recommandations_neuro(prediction)
        for r in recommandations:
            st.markdown(f'<div class="rx-item">→ {r}</div>', unsafe_allow_html=True)

        pdf_label = "No Tumor" if prediction == 0 else "Tumor"
        pdf_bytes = generer_pdf_neuro(pdf_label, confidence, recommandations)
        st.download_button(
            label="📥 Télécharger le Rapport IRM (PDF)",
            data=pdf_bytes,
            file_name="Rapport_IRM_VET_AI.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC BIOLOGIQUE
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">🔬 Paramètres Biologiques de l\'Animal</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; font-size:0.85rem; margin-bottom:20px;">Saisissez les résultats d\'analyse biologique pour obtenir un diagnostic automatique.</p>', unsafe_allow_html=True)

    model_bio = load_bio_model()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🐾 Âge (années)", min_value=0, max_value=30, value=5, step=1)
        pression = st.number_input("💓 Pression Artérielle (mmHg)", min_value=60, max_value=250, value=130, step=1)
        densite = st.number_input("🧪 Densité Urinaire", min_value=1.000, max_value=1.060, value=1.020, step=0.001, format="%.3f")
    with col2:
        creatinine = st.number_input("🩸 Créatinine (mg/L)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
        uree = st.number_input("⚗️ Urée (g/L)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        hemoglobine = st.number_input("🔴 Hémoglobine (g/dL)", min_value=0.0, max_value=25.0, value=12.0, step=0.1)

    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-val">RFC</div>
            <div class="metric-label">Algorithme</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">6</div>
            <div class="metric-label">Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">200</div>
            <div class="metric-label">Arbres</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Lancer le Diagnostic Biologique", use_container_width=True):
        params = [age, pression, densite, creatinine, uree, hemoglobine]

        with st.spinner("🔬 Analyse des paramètres biologiques en cours..."):
            prediction_bio, confidence_bio = predict_bio(model_bio, params)

        st.markdown('<p class="section-header">🎯 Résultat Diagnostique</p>', unsafe_allow_html=True)

        if prediction_bio == 0:
            st.markdown(f"""
            <div class="card-sain">
                <p class="card-title">✅ PARAMÈTRES NORMAUX — ANIMAL SAIN</p>
                <p class="card-sub">Aucune anomalie biologique significative détectée</p>
                <div class="conf-bar-container">
                    <div class="conf-bar-fill-green" style="width:{confidence_bio:.1f}%"></div>
                </div>
                <p style="color:#10B981; font-size:0.8rem; margin:0; font-family:'DM Mono',monospace;">
                    Confiance algorithmique : {confidence_bio:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-malade">
                <p class="card-title">⚠️ ANOMALIE DÉTECTÉE — CONSULTATION REQUISE</p>
                <p class="card-sub">Paramètres biologiques en dehors des normes vétérinaires</p>
                <div class="conf-bar-container">
                    <div class="conf-bar-fill-red" style="width:{confidence_bio:.1f}%"></div>
                </div>
                <p style="color:#EF4444; font-size:0.8rem; margin:0; font-family:'DM Mono',monospace;">
                    Confiance algorithmique : {confidence_bio:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">💊 Orientations Cliniques</p>', unsafe_allow_html=True)
        recommandations_bio = obtenir_recommandations_bio(prediction_bio, hemoglobine, creatinine, uree)
        for r in recommandations_bio:
            st.markdown(f'<div class="rx-item">→ {r}</div>', unsafe_allow_html=True)

        pdf_bytes_bio = generer_pdf_bio(prediction_bio, confidence_bio, recommandations_bio,
                                         age, pression, densite, creatinine, uree, hemoglobine)
        st.download_button(
            label="📥 Télécharger le Rapport Biologique (PDF)",
            data=pdf_bytes_bio,
            file_name="Rapport_Biologique_VET_AI.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    VET-AI PLATFORM • ENSV • Mini-Projet IA en Recherche Vétérinaire<br>
    ⚠️ Outil d'aide au diagnostic — Ne remplace pas l'expertise clinique vétérinaire
</div>
""", unsafe_allow_html=True)

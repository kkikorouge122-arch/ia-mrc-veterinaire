import streamlit as st
from model_utils import (
    load_medical_cnn, predict_mri_image,
    load_bio_model, predict_bio,
    obtenir_recommandations_neuro, obtenir_recommandations_bio,
    generer_pdf_neuro, generer_pdf_bio
)
from PIL import Image
import pandas as pd
import numpy as np

st.set_page_config(page_title="VET-AI | Diagnostic Platform", page_icon="🧬", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace !important; background-color: #020817 !important; }
    .stApp { background: #020817; }
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid #334155; border-radius: 24px;
        padding: 40px 30px; text-align: center; margin-bottom: 32px;
        position: relative; overflow: hidden;
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important; font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #10B981, #06B6D4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; letter-spacing: -1px;
    }
    .hero-subtitle { color: #64748B; font-size: 0.85rem; letter-spacing: 3px; text-transform: uppercase; margin-top: 8px; }
    .hero-badge {
        display: inline-block; background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4); color: #818CF8;
        padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; margin-top: 16px; letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #0F172A; border: 1px solid #1E293B; border-radius: 16px; padding: 6px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #64748B; border-radius: 12px;
        font-family: 'DM Mono', monospace; font-size: 0.85rem; padding: 10px 20px; border: none;
    }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366F1, #4F46E5) !important; color: white !important; }
    .section-header {
        font-family: 'Syne', sans-serif; color: #E2E8F0; font-size: 1.1rem; font-weight: 600;
        border-left: 3px solid #6366F1; padding-left: 12px; margin: 24px 0 16px 0;
    }
    .card-sain {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
        border: 1px solid rgba(16,185,129,0.4); padding: 24px; border-radius: 20px;
        color: white; margin: 16px 0;
    }
    .card-malade {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.1));
        border: 1px solid rgba(239,68,68,0.4); padding: 24px; border-radius: 20px;
        color: white; margin: 16px 0;
    }
    .card-title { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; margin: 0 0 6px 0; }
    .card-sub { font-size: 0.85rem; opacity: 0.8; margin: 0; }
    .conf-bar-container { background: #1E293B; border-radius: 8px; height: 8px; margin: 12px 0 4px 0; overflow: hidden; }
    .conf-bar-green { height: 100%; background: linear-gradient(90deg, #10B981, #34D399); border-radius: 8px; }
    .conf-bar-red { height: 100%; background: linear-gradient(90deg, #EF4444, #F87171); border-radius: 8px; }
    .rx-item {
        background: #0F172A; border: 1px solid #1E293B; border-left: 3px solid #6366F1;
        padding: 14px 18px; border-radius: 0 12px 12px 0; margin-bottom: 10px;
        color: #CBD5E1; font-size: 0.88rem; line-height: 1.6;
    }
    .stat-card {
        background: #0F172A; border: 1px solid #1E293B; border-radius: 14px;
        padding: 16px; text-align: center; margin-bottom: 8px;
    }
    .stat-val { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #6366F1; }
    .stat-label { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: white !important; border: none !important; padding: 14px 20px !important;
        border-radius: 14px !important; font-weight: 600 !important; width: 100% !important; margin-top: 16px !important;
    }
    .footer {
        text-align: center; color: #334155; font-size: 0.75rem;
        margin-top: 40px; padding-top: 20px; border-top: 1px solid #1E293B; letter-spacing: 1px;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# HERO HEADER
st.markdown("""
<div class="hero-header">
    <p class="hero-subtitle">École Nationale Supérieure Vétérinaire • IA Médicale</p>
    <h1 class="hero-title">🧬 VET-AI Platform</h1>
    <p style="color:#64748B; font-size:0.9rem; margin-top:10px; font-family:'DM Mono',monospace;">
        Système de Diagnostic Assisté par Intelligence Artificielle
    </p>
    <span class="hero-badge">Random Forest Classifier • v2.0 • Accuracy 96.9% IRM | 100% Bio</span>
</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs([
    "🧠  Analyse IRM Cérébrale",
    "🔬  Diagnostic Biologique",
    "📊  Exploration du Dataset"
])

# ════════════════════════════════════════════
# TAB 1 — IRM
# ════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">📁 Téléversement du Cliché IRM</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; font-size:0.85rem; margin-bottom:16px;">Importez une image IRM cérébrale au format JPG ou PNG.</p>', unsafe_allow_html=True)

    model_irm = load_medical_cnn()
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="irm_uploader", label_visibility="collapsed")

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
                <p class="card-sub">Aucune lésion expansive détectée</p>
                <div class="conf-bar-container"><div class="conf-bar-green" style="width:{confidence:.1f}%"></div></div>
                <p style="color:#10B981; font-size:0.8rem; margin:0;">Confiance : {confidence:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-malade">
                <p class="card-title">🚨 ANOMALIE — TUMEUR DÉTECTÉE</p>
                <p class="card-sub">Suspicion de processus tumoral — Consultation urgente</p>
                <div class="conf-bar-container"><div class="conf-bar-red" style="width:{confidence:.1f}%"></div></div>
                <p style="color:#EF4444; font-size:0.8rem; margin:0;">Confiance : {confidence:.2f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">💊 Orientations Cliniques</p>', unsafe_allow_html=True)
        recommandations = obtenir_recommandations_neuro(prediction)
        for r in recommandations:
            st.markdown(f'<div class="rx-item">→ {r}</div>', unsafe_allow_html=True)

        pdf_bytes = generer_pdf_neuro("No Tumor" if prediction == 0 else "Tumor", confidence, recommandations)
        st.download_button("📥 Télécharger le Rapport IRM (PDF)", data=pdf_bytes,
                           file_name="Rapport_IRM_VET_AI.pdf", mime="application/pdf", use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — BIOLOGIQUE
# ════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">🔬 Paramètres Biologiques de l\'Animal</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; font-size:0.85rem; margin-bottom:20px;">Saisissez les résultats d\'analyse biologique.</p>', unsafe_allow_html=True)

    model_bio = load_bio_model()

    col1, col2 = st.columns(2)
    with col1:
        age        = st.number_input("🐾 Âge (années)", min_value=0, max_value=30, value=5, step=1)
        pression   = st.number_input("💓 Pression Artérielle (mmHg)", min_value=60, max_value=250, value=130, step=1)
        densite    = st.number_input("🧪 Densité Urinaire", min_value=1.000, max_value=1.060, value=1.020, step=0.001, format="%.3f")
    with col2:
        creatinine = st.number_input("🩸 Créatinine (mg/L)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
        uree       = st.number_input("⚗️ Urée (g/L)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        hemoglobine= st.number_input("🔴 Hémoglobine (g/dL)", min_value=0.0, max_value=25.0, value=12.0, step=0.1)

    if st.button("⚡ Lancer le Diagnostic Biologique", use_container_width=True):
        params = [age, pression, densite, creatinine, uree, hemoglobine]
        with st.spinner("🔬 Analyse en cours..."):
            prediction_bio, confidence_bio = predict_bio(model_bio, params)

        st.markdown('<p class="section-header">🎯 Résultat Diagnostique</p>', unsafe_allow_html=True)
        if prediction_bio == 0:
            st.markdown(f"""
            <div class="card-sain">
                <p class="card-title">✅ PARAMÈTRES NORMAUX — ANIMAL SAIN</p>
                <p class="card-sub">Aucune anomalie biologique significative</p>
                <div class="conf-bar-container"><div class="conf-bar-green" style="width:{confidence_bio:.1f}%"></div></div>
                <p style="color:#10B981; font-size:0.8rem; margin:0;">Confiance : {confidence_bio:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-malade">
                <p class="card-title">⚠️ ANOMALIE DÉTECTÉE — CONSULTATION REQUISE</p>
                <p class="card-sub">Paramètres biologiques en dehors des normes vétérinaires</p>
                <div class="conf-bar-container"><div class="conf-bar-red" style="width:{confidence_bio:.1f}%"></div></div>
                <p style="color:#EF4444; font-size:0.8rem; margin:0;">Confiance : {confidence_bio:.2f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">💊 Orientations Cliniques</p>', unsafe_allow_html=True)
        recs_bio = obtenir_recommandations_bio(prediction_bio, hemoglobine, creatinine, uree)
        for r in recs_bio:
            st.markdown(f'<div class="rx-item">→ {r}</div>', unsafe_allow_html=True)

        pdf_bytes_bio = generer_pdf_bio(prediction_bio, confidence_bio, recs_bio,
                                        age, pression, densite, creatinine, uree, hemoglobine)
        st.download_button("📥 Télécharger le Rapport Biologique (PDF)", data=pdf_bytes_bio,
                           file_name="Rapport_Bio_VET_AI.pdf", mime="application/pdf", use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — EXPLORATION DATASET
# ════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">📊 Exploration du Dataset Biologique</p>', unsafe_allow_html=True)

    try:
        df = pd.read_csv("dataset_transposed.csv")

        # ── STATISTIQUES GLOBALES ──────────────────────────
        n_total  = len(df)
        n_sain   = len(df[df['Diagnostic'] == 'Sain'])
        n_malade = len(df[df['Diagnostic'] == 'Malade'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{n_total}</div><div class="stat-label">Total Animaux</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#10B981">{n_sain}</div><div class="stat-label">Animaux Sains</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#EF4444">{n_malade}</div><div class="stat-label">Animaux Malades</div></div>', unsafe_allow_html=True)

        # ── FILTRE ────────────────────────────────────────
        st.markdown('<p class="section-header">🔍 Filtrer par Diagnostic</p>', unsafe_allow_html=True)
        filtre = st.radio("", ["Tous", "Sain", "Malade"], horizontal=True, label_visibility="collapsed")

        if filtre == "Tous":
            df_filtre = df
        else:
            df_filtre = df[df['Diagnostic'] == filtre]

        st.markdown(f'<p style="color:#64748B; font-size:0.85rem;">{len(df_filtre)} résultats affichés</p>', unsafe_allow_html=True)

        # ── TABLEAU INTERACTIF ────────────────────────────
        st.markdown('<p class="section-header">📋 Tableau des Données</p>', unsafe_allow_html=True)
        st.dataframe(
            df_filtre.style.apply(
                lambda row: ['background-color: rgba(16,185,129,0.1)' if row['Diagnostic'] == 'Sain'
                             else 'background-color: rgba(239,68,68,0.1)'] * len(row),
                axis=1
            ),
            use_container_width=True,
            height=300
        )

        # ── STATISTIQUES PAR GROUPE ───────────────────────
        st.markdown('<p class="section-header">📈 Statistiques par Groupe</p>', unsafe_allow_html=True)
        stats = df.groupby('Diagnostic')[['Age','Pression_Arterielle','Creatinine_mg_L',
                                          'Uree_g_L','Hemoglobine_g_dL']].mean().round(2)
        st.dataframe(stats, use_container_width=True)

        # ── GRAPHIQUES ────────────────────────────────────
        st.markdown('<p class="section-header">📊 Distribution des Paramètres</p>', unsafe_allow_html=True)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.facecolor'] = '#0F172A'
        matplotlib.rcParams['axes.facecolor']   = '#1E293B'
        matplotlib.rcParams['text.color']        = '#E2E8F0'
        matplotlib.rcParams['axes.labelcolor']   = '#94A3B8'
        matplotlib.rcParams['xtick.color']       = '#64748B'
        matplotlib.rcParams['ytick.color']       = '#64748B'
        matplotlib.rcParams['axes.edgecolor']    = '#334155'
        matplotlib.rcParams['grid.color']        = '#1E293B'

        params_plot = ['Hemoglobine_g_dL', 'Creatinine_mg_L', 'Uree_g_L', 'Pression_Arterielle']
        labels_plot = ['Hémoglobine (g/dL)', 'Créatinine (mg/L)', 'Urée (g/L)', 'Pression (mmHg)']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#0F172A')
        axes = axes.flatten()

        df_sain   = df[df['Diagnostic'] == 'Sain']
        df_malade = df[df['Diagnostic'] == 'Malade']

        for i, (param, label) in enumerate(zip(params_plot, labels_plot)):
            axes[i].hist(df_sain[param],   bins=20, alpha=0.7, color='#10B981', label='Sain',   edgecolor='none')
            axes[i].hist(df_malade[param], bins=20, alpha=0.7, color='#EF4444', label='Malade', edgecolor='none')
            axes[i].set_title(label, fontsize=11, fontweight='bold', color='#E2E8F0')
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.2)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        # ── BOXPLOT ───────────────────────────────────────
        st.markdown('<p class="section-header">📦 Comparaison Sain vs Malade</p>', unsafe_allow_html=True)

        param_select = st.selectbox("Choisir un paramètre :", params_plot,
                                    format_func=lambda x: dict(zip(params_plot, labels_plot))[x])

        fig2, ax = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')

        data_sain   = df_sain[param_select].dropna()
        data_malade = df_malade[param_select].dropna()

        bp = ax.boxplot([data_sain, data_malade], labels=['Sain', 'Malade'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2))
        bp['boxes'][0].set_facecolor('#10B981')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#EF4444')
        bp['boxes'][1].set_alpha(0.7)

        ax.set_title(f'Distribution — {dict(zip(params_plot, labels_plot))[param_select]}',
                     fontsize=12, fontweight='bold', color='#E2E8F0')
        ax.grid(True, alpha=0.2, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig2)
        plt.close()

    except FileNotFoundError:
        st.warning("Dataset non trouvé. Assurez-vous que dataset_transposed.csv est présent.")

# FOOTER
st.markdown("""
<div class="footer">
    VET-AI PLATFORM • ENSV Alger • Mini-Projet IA en Recherche Vétérinaire<br>
    ⚠️ Outil d'aide au diagnostic — Ne remplace pas l'expertise clinique vétérinaire
</div>
""", unsafe_allow_html=True)

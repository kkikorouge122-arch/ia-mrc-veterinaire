import streamlit as st
from model_utils import load_and_train_model, generate_importance_plot, obtenir_traitement
import numpy as np

# Configuration de la page Premium
st.set_page_config(
    page_title="VET-AI | Diagnostic MRC", 
    page_icon="⚡", 
    layout="centered"
)

# Injection CSS pour un design professionnel haut de gamme et des couleurs vives
st.markdown("""
<style>
    /* Style global et police */
    @import url('https://googleapis.com');
    html, body, [data-testid="stWidgetLabel"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Design des boutons principaux */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00D2FF 0%, #0066FF 100%);
        color: white;
        border: none;
        padding: 14px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }
    
    /* Cartes personnalisées pour l'affichage des résultats */
    .card-sain {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 20px; border-radius: 16px; color: white; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    .card-precoce {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        padding: 20px; border-radius: 16px; color: white; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    }
    .card-avance {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        padding: 20px; border-radius: 16px; color: white; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    
    /* Traitement et recommandations listes */
    .rx-item {
        background-color: #1E293B;
        border-left: 4px solid #00D2FF;
        padding: 12px 16px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_base64=True)

# Header de l'application
st.markdown("<h1 style='text-align: center; color: #00D2FF; font-weight: 800; font-size: 2.5rem; margin-bottom: 5px;'>🐾 VET-AI CLINIC</h1>", unsafe_base64=True)
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 30px;'>Plateforme décisionnelle de pointe • Stadification de la MRC Féline</p>", unsafe_base64=True)

# Chargement de l'IA en arrière-plan
@st.cache_resource
def get_cached_model():
    return load_and_train_model()

model, feature_names = get_cached_model()

# Zone de saisie utilisateur
st.markdown("<h3 style='color: #F1F5F9; border-bottom: 2px solid #334155; padding-bottom: 8px;'>📋 Paramètres du Patient</h3>", unsafe_base64=True)

# Layout adaptatif en deux colonnes
col1, col2 = st.columns(2)

with col1:
    age = st.slider("📆 Âge du félin (années)", 1, 22, 10)
    pa = st.slider("💓 Pression Artérielle Systolique (mmHg)", 90, 220, 130)
    du = st.selectbox("🧪 Densité Urinaire (DU)", [1.010, 1.020, 1.035, 1.045], index=2)

with col2:
    creat = st.number_input("🩸 Créatinine Sérique (mg/L)", min_value=5.0, max_value=80.0, value=14.0, step=1.0)
    uree = np.round(st.number_input("⚗️ Urée Sérique (g/L)", min_value=0.1, max_value=4.0, value=0.4, step=0.1), 2)
    hemo = st.number_input("📉 Hémoglobine (g/dL)", min_value=5.0, max_value=18.0, value=12.0, step=0.5)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_base64=True)

# Bouton de traitement
if st.button("🧬 LANCER L'ANALYSE PRÉDICTIVE", use_container_width=True):
    # Simulation d'un loader haut de gamme
    with st.spinner("Calcul des probabilités cliniques..."):
        input_data = [[age, pa, du, creat, uree, hemo]]
        prediction = model.predict(input_data)[0]
        
    st.markdown("<h3 style='color: #F1F5F9; border-bottom: 2px solid #334155; padding-bottom: 8px; margin-top: 30px;'>🎯 Verdict de l'Intelligence Artificielle</h3>", unsafe_base64=True)
    
    # Affichage personnalisé avec les cartes CSS de couleurs vives
    if prediction == 0:
        st.markdown("""
        <div class='card-sain'>
            <h4 style='margin:0; font-weight:800; font-size:1.3rem;'>🎉 PATIENT SAIN / CONTRÔLE</h4>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Le profil métabolique ne présente aucun marqueur de dysfonctionnement rénal chronique.</p>
        </div>
        """, unsafe_base64=True)
    elif prediction == 1:
        st.markdown("""
        <div class='card-precoce'>
            <h4 style='margin:0; font-weight:800; font-size:1.3rem;'>⚠️ MRC STADE PRÉCOCE (IRIS 1-2)</h4>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Alerte de filtration. Les capacités de concentration du rein commencent à s'altérer de manière critique.</p>
        </div>
        """, unsafe_base64=True)
    else:
        st.markdown("""
        <div class='card-avance'>
            <h4 style='margin:0; font-weight:800; font-size:1.3rem;'>🚨 MRC STADE AVANCÉ (IRIS 3-4)</h4>
            <p style='margin:5px 0 0 0; opacity:0.9;'>Urgence médicale. Forte dégradation fonctionnelle des néphrons associée à un risque majeur d'urémie clinique.</p>
        </div>
        """, unsafe_base64=True)
        
    # Recommandations Médicales
    st.markdown("<h4 style='color: #00D2FF; margin-top:20px;'>💊 Protocole Médical Suggéré</h4>", unsafe_base64=True)
    traitements = obtenir_traitement(prediction, pa)
    for t in traitements:
        st.markdown(f"<div class='rx-item'>{t}</div>", unsafe_base64=True)
        
    # Graphique de validation scientifique
    st.markdown("<h4 style='color: #00D2FF; margin-top:30px;'>📊 Poids Scientifique des Variables (Explicabilité)</h4>", unsafe_base64=True)
    fig = generate_importance_plot(model, feature_names)
    st.pyplot(fig)

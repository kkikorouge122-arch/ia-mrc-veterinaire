import streamlit as st
from model_utils import load_medical_cnn, predict_mri_image, obtenir_recommandations_neuro, generer_pdf_neuro
from PIL import Image

# Configuration de la page de Neuro-Oncologie
st.set_page_config(page_title="VET-AI | Neuro-Vision", page_icon="🧠", layout="centered")

# Design CSS Premium à Couleurs Vives
st.markdown("""
<style>
    @import url('https://googleapis.com');
    html, body, [data-testid="stWidgetLabel"] { font-family: 'Inter', sans-serif !important; }
    
    /* Boutons de téléchargement */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #FF9900 0%, #FF5500 100%) !important;
        color: white !important; border: none !important; padding: 14px 20px !important;
        border-radius: 12px !important; font-weight: 600 !important; font-size: 16px !important;
        box-shadow: 0 4px 15px rgba(255, 153, 0, 0.3) !important;
    }
    
    /* Cartes cliniques colorées */
    .card-normal {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 20px; border-radius: 16px; color: white; margin-bottom: 20px;
    }
    .card-tumor {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        padding: 20px; border-radius: 16px; color: white; margin-bottom: 20px;
    }
    
    /* Boîtes de recommandations */
    .rx-item {
        background-color: #1E293B; border-left: 4px solid #00D2FF;
        padding: 14px 16px; border-radius: 0 12px 12px 0; margin-bottom: 12px;
    }
    .rx-text {
        color: #FFFFFF !important; font-size: 14.5px !important;
        font-weight: 600 !important; margin: 0 !important; padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #00D2FF; font-weight: 800; font-size: 2.3rem;'>🧠 VET-AI NEURO-VISION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 30px;'>Pipeline Deep Learning • Classification Automatique de Tumeurs Cerebrales (IRM)</p>", unsafe_allow_html=True)

# Chargement à froid du modèle (Évite les blocages de cache mémoire)
model_instance = load_medical_cnn()

st.markdown("<h3 style='color: #F1F5F9; border-bottom: 2px solid #334155; padding-bottom: 8px;'>📁 Televersement du Cliche IRM</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Selectionnez une image de scanner ou d'IRM cerebrale (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Cliché IRM chargé avec succès pour l'analyse", use_container_width=True)
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # Inférence automatique pure
    with st.spinner("Analyse des densites de l'image en cours..."):
        prediction, confidence = predict_mri_image(model_instance, image)
        
    st.markdown("<h3 style='color: #F1F5F9; border-bottom: 2px solid #334155; padding-bottom: 8px; margin-top: 20px;'>🎯 Verdict Diagnostique du Reseau (CNN)</h3>", unsafe_allow_html=True)
    
    if prediction == 0:
        st.markdown(f"""
        <div class='card-normal'>
            <h4 style='margin:0; font-weight:800; color: white;'>🎉 EXAMEN NORMAL : CLASSIFIE SAIN</h4>
            <p style='margin:5px 0 0 0; color: white; opacity:0.9;'>Certitude algorithmique : {confidence:.2f}% • Aucune lesion expansive detectee.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='card-tumor'>
            <h4 style='margin:0; font-weight:800; color: white;'>🚨 ANOMALIE DETECTEE : TUMEUR CEREBRALE</h4>
            <p style='margin:5px 0 0 0; color: white; opacity:0.9;'>Certitude algorithmique : {confidence:.2f}% • Suspicion immediate de processus tumoral.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h4 style='color: #00D2FF;'>💊 Orientations Cliniques Immediate</h4>", unsafe_allow_html=True)
    recommandations = obtenir_recommandations_neuro(prediction)
    for r in recommandations:
        st.markdown(f"<div class='rx-item'><p class='rx-text'>{r}</p></div>", unsafe_allow_html=True)
        
    st.markdown("<h4 style='color: #00D2FF; margin-top:30px;'>💾 Archivage Academique</h4>", unsafe_allow_html=True)
    pdf_label = "No Tumor" if prediction == 0 else "Tumor"
    
    # Génération du PDF converti en type binaire exploitable par Streamlit
    pdf_output = generer_pdf_neuro(pdf_label, confidence, recommandations)
    pdf_bytes = bytes(pdf_output)
    
    st.download_button(
        label="📥 Télécharger le Rapport d'Imagerie Officiel (PDF)",
        data=pdf_bytes,
        file_name="Rapport_NeuroVision_Patient.pdf",
        mime="application/pdf",
        use_container_width=True
    )

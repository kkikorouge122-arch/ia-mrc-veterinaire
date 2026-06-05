import numpy as np
import pickle
import os
from fpdf import FPDF

MODEL_PATH = "brain_tumor_model.pkl"
IMG_SIZE = (64, 64)

def extraire_features(pil_image):
    # Conversion en niveaux de gris et redimensionnement
    img_gray = pil_image.convert("L").resize(IMG_SIZE)
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    
    # Extraction de l'histogramme (16 bins)
    hist, _ = np.histogram(arr, bins=16, range=(0, 1))
    hist = hist / hist.sum()
    
    # Statistiques globales du tableau de pixels
    mean      = np.mean(arr)
    std       = np.std(arr)
    variance  = np.var(arr)
    max_val   = np.max(arr)
    min_val   = np.min(arr)
    
    # Ratios de brillance et de zones sombres
    ratio_bright = np.sum(arr > 0.75) / arr.size
    ratio_dark   = np.sum(arr < 0.20) / arr.size
    
    # Calcul de l'asymétrie gauche/droite de l'IRM
    left  = arr[:, :IMG_SIZE[1]//2]
    right = arr[:, IMG_SIZE[1]//2:]
    asymmetry = abs(np.mean(left) - np.mean(right))
    
    stats = np.array([mean, std, variance, max_val, min_val,
                      ratio_bright, ratio_dark, asymmetry])
    
    return np.concatenate([hist, stats])

def load_medical_cnn():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modele introuvable. Lancez d'abord: python train_model.py")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_mri_image(model, pil_image):
    features = extraire_features(pil_image).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction]) * 100
    return prediction, confidence

def obtenir_recommandations_neuro(prediction):
    if prediction == 0:
        return [
            "Absence d'anomalie de densite visible ou de masse intracranienne occupante.",
            "Planification d'un suivi neurologique standard de controle si les symptomes persistent."
        ]
    else:
        return [
            "ALERTE : Presence d'une masse tissulaire evoquant un processus neoplasique.",
            "Instaurer une corticotherapie anti-oedemateuse immediate (ex: Dexamethasone).",
            "Orienter d'urgence vers un service de neurochirurgie veterinaire."
        ]

def nettoyer_texte(texte):
    remplacements = {
        'é': 'e', 'è': 'e', 'à': 'a', 'ù': 'u', 'ç': 'c',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u', 
        '’': "'", '°': 'o'
    }
    for accent, lettre in remplacements.items():
        texte = texte.replace(accent, lettre)
    return texte.encode('latin-1', 'replace').decode('latin-1')

def generer_pdf_neuro(prediction_label, confidence, recommandations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    
    # En-tête
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 102, 255)
    titre = nettoyer_texte("RAPPORT D'IMAGERIE INFORMATISÉ - VET-AI")
    pdf.cell(180, 10, titre, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Sous-titre
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(180, 5, "Analyse Automatique par Apprentissage Statistique", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(15, 35, 195, 35)
    pdf.ln(10)
    
    # Section Verdict
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 10, "1. Verdict :", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 11)
    if prediction_label.lower() == "tumor":
        pdf.set_text_color(220, 50, 50)
        label_propre = "TUMEUR DETECTEE"
    else:
        pdf.set_text_color(16, 185, 129)
        label_propre = "SUJET SAIN"
        
    verdict_text = nettoyer_texte(f"CLASSIFICATION : {label_propre} (Confiance : {confidence:.2f}%)")
    pdf.cell(180, 8, verdict_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section Recommandations
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 10, "2. Orientations cliniques :", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    for r in recommandations:
        ligne = nettoyer_texte(f"* {r}")
        pdf.multi_cell(180, 6, ligne)
        
    pdf.ln(15)
    
    # Clause de non-responsabilité (Avertissement complet)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    avertissement = nettoyer_texte(
        "Avertissement : Ce rapport est genere automatiquement par un systeme d'intelligence artificielle "
        "a but educatif et d'assistance. Il ne remplace en aucun cas l'expertise clinique d'un radiologue "
        "ou d'un specialiste en medecine veterinaire. Toute decision therapeutique doit etre validee par un professionnel."
    )
    pdf.multi_cell(180, 5, avertissement)
    
    # Retourne le PDF sous forme d'octets binaires string pour Streamlit
    return pdf.output(dest='S')

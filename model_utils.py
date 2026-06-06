import numpy as np
import pickle
import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos


MODEL_PATH_IRM = "brain_tumor_model.pkl"
MODEL_PATH_BIO = "bio_model.pkl"
IMG_SIZE = (64, 64)


# ═══════════════════════════════════════════════════
#  MODULE 1 — ANALYSE IRM CÉRÉBRALE
# ═══════════════════════════════════════════════════

def extraire_features(pil_image):
    img_gray = pil_image.convert("L").resize(IMG_SIZE)
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=16, range=(0, 1))
    hist = hist / hist.sum()
    mean         = np.mean(arr)
    std          = np.std(arr)
    variance     = np.var(arr)
    max_val      = np.max(arr)
    min_val      = np.min(arr)
    ratio_bright = np.sum(arr > 0.75) / arr.size
    ratio_dark   = np.sum(arr < 0.20) / arr.size
    left         = arr[:, :IMG_SIZE[1]//2]
    right        = arr[:, IMG_SIZE[1]//2:]
    asymmetry    = abs(np.mean(left) - np.mean(right))
    stats = np.array([mean, std, variance, max_val, min_val,
                      ratio_bright, ratio_dark, asymmetry])
    return np.concatenate([hist, stats])


def load_medical_cnn():
    if not os.path.exists(MODEL_PATH_IRM):
        raise FileNotFoundError("Modele IRM introuvable. Lancez: python train_model.py")
    with open(MODEL_PATH_IRM, 'rb') as f:
        return pickle.load(f)


def predict_mri_image(model, pil_image):
    features = extraire_features(pil_image).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction]) * 100
    return prediction, confidence


def obtenir_recommandations_neuro(prediction):
    if prediction == 0:
        return [
            "Aucune anomalie de densite visible ni masse intracranienne occupante detectee.",
            "Suivi neurologique standard recommande si les symptomes cliniques persistent.",
            "Reevaluation IRM conseillee dans 6 mois en cas de signes neurologiques."
        ]
    else:
        return [
            "ALERTE : Masse tissulaire evoquant un processus neoplasique (Gliome/Meningiome).",
            "Instaurer une corticotherapie anti-oedemateuse immediate (ex: Dexamethasone).",
            "Orienter d'urgence vers un service de neurochirurgie veterinaire specialise.",
            "Bilan d'extension recommande : scanner thoracique et echographie abdominale."
        ]


# ═══════════════════════════════════════════════════
#  MODULE 2 — DIAGNOSTIC BIOLOGIQUE
# ═══════════════════════════════════════════════════

def load_bio_model():
    if not os.path.exists(MODEL_PATH_BIO):
        raise FileNotFoundError("Modele biologique introuvable. Lancez: python train_bio_model.py")
    with open(MODEL_PATH_BIO, 'rb') as f:
        return pickle.load(f)

def predict_bio(model, features):
    import numpy as np
    import pandas as pd

    # 1. Extraction des valeurs numériques de manière brute
    if isinstance(features, dict):
        # On extrait les valeurs dans l'ordre du dictionnaire
        vals = list(features.values())
    elif isinstance(features, pd.DataFrame):
        vals = features.iloc[0].tolist()
    else:
        vals = list(features)

    # 2. Sécurisation : on force le format à exactement 5 caractéristiques
    # Le modèle attend : Age, Pression_Arterielle, Densite_Urinaire, Creatinine_mg_L, Uree_g_L
    vals = vals[:5]  # On ne garde que les 5 premiers éléments
    
    # 3. Conversion en tableau 2D pour scikit-learn
    input_data = np.array(vals).reshape(1, -1)

    # 4. Prédiction et calcul du score de confiance
    prediction = int(model.predict(input_data)[0])
    
    try:
        # On extrait la probabilité de la classe prédite
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction])
    except Exception:
        confidence = 1.0

    return prediction, confidence




def obtenir_recommandations_bio(prediction, hemoglobine, creatinine, uree):
    if prediction == 0:
        return [
            "Parametres biologiques dans les normes veterinaires — Etat general satisfaisant.",
            "Maintenir un suivi biologique annuel de routine.",
            "Hydratation et alimentation equilibree recommandees pour la prevention."
        ]
    else:
        recs = ["ANOMALIE BIOLOGIQUE DETECTEE — Consultation veterinaire urgente requise."]
        if hemoglobine < 10.0:
            recs.append(f"Anemie suspectee (Hb = {hemoglobine:.1f} g/dL) — Bilan hematologique complet indique.")
        if creatinine > 25.0:
            recs.append(f"Hypercreatininemie (Creat = {creatinine:.1f} mg/L) — Suspicion d'insuffisance renale.")
        if uree > 0.8:
            recs.append(f"Hyperuremie (Uree = {uree:.2f} g/L) — Controle de la fonction renale necessaire.")
        recs.append("Echographie renale et bilan urologique recommandes en urgence.")
        return recs


# ═══════════════════════════════════════════════════
#  MODULE 3 — GENERATION PDF
# ═══════════════════════════════════════════════════

def nettoyer_texte(texte):
    remplacements = {
        'e': 'e', 'e': 'e', 'a': 'a', 'u': 'u', 'c': 'c',
        'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
        '\u2019': "'", '\u00e9': 'e', '\u00e8': 'e', '\u00e0': 'a',
        '\u00f9': 'u', '\u00e7': 'c', '\u00e2': 'a', '\u00ea': 'e',
        '\u00ee': 'i', '\u00f4': 'o', '\u00fb': 'u', '\u00b0': 'o'
    }
    for accent, lettre in remplacements.items():
        texte = texte.replace(accent, lettre)
    return texte.encode('latin-1', 'replace').decode('latin-1')


def _entete_pdf(pdf, titre, sous_titre):
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(180, 10, nettoyer_texte(titre), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(180, 5, nettoyer_texte(sous_titre), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(15, pdf.get_y() + 3, 195, pdf.get_y() + 3)
    pdf.ln(10)


def generer_pdf_neuro(prediction_label, confidence, recommandations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    _entete_pdf(pdf, "RAPPORT D'IMAGERIE IRM - VET-AI",
                "Analyse Automatique par Apprentissage Automatique | Module Neuro-Oncologie")

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 8, "1. Resultat Diagnostique :", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 11)
    if prediction_label.lower() == "tumor":
        pdf.set_text_color(200, 40, 40)
        label = "TUMEUR CEREBRALE DETECTEE"
    else:
        pdf.set_text_color(20, 160, 100)
        label = "EXAMEN NORMAL - SUJET SAIN"
    pdf.cell(180, 8, nettoyer_texte(f"Classification : {label} (Confiance : {confidence:.2f}%)"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 8, "2. Orientations Cliniques :", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in recommandations:
        pdf.multi_cell(180, 6, nettoyer_texte(f"  * {r}"))
    pdf.ln(10)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(180, 5, nettoyer_texte(
        "Avertissement : Ce rapport est genere automatiquement a titre educatif. "
        "Il ne remplace pas l'expertise d'un radiologue ou neurologue veterinaire qualifie."
    ))
    return bytes(pdf.output())


def generer_pdf_bio(prediction, confidence, recommandations,
                    age, pression, densite, creatinine, uree, hemoglobine):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    _entete_pdf(pdf, "RAPPORT BIOLOGIQUE VETERINAIRE - VET-AI",
                "Diagnostic Automatique par Parametres Biologiques | Module Bio-Clinique")

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 8, "1. Parametres Biologiques Analyses :", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = [
        ("Age", f"{age} ans"),
        ("Pression Arterielle", f"{pression} mmHg"),
        ("Densite Urinaire", f"{densite:.3f}"),
        ("Creatinine", f"{creatinine:.1f} mg/L"),
        ("Uree", f"{uree:.2f} g/L"),
        ("Hemoglobine", f"{hemoglobine:.1f} g/dL"),
    ]
# Exemple de structure standard FPDF sans new_x / new_y
# Remplacez la boucle d'affichage des paramètres par ceci :
for nom, valeur u in liste_parametres:
    pdf.cell(90, 7, nettoyer_texte(f"  {nom} :"), 0, 0)
    pdf.cell(90, 7, nettoyer_texte(f"{valeur}"), 0, 1) # Le '1' à la fin indique un retour à la ligne automatique


    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 8, "2. Resultat Diagnostique :", new_x="lmargin", new_y="next")
    pdf.set_font("Helvetica", "B", 11)
    if prediction == 1:
        pdf.set_text_color(200, 40, 40)
        label = "ANOMALIE BIOLOGIQUE DETECTEE"
    else:
        pdf.set_text_color(20, 160, 100)
        label = "PARAMETRES NORMAUX - ANIMAL SAIN"
    pdf.cell(180, 8, nettoyer_texte(f"Classification : {label} (Confiance : {confidence:.2f}%)"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 8, "3. Orientations Cliniques :", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in recommandations:
        pdf.multi_cell(180, 6, nettoyer_texte(f"  * {r}"))
    pdf.ln(10)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(180, 5, nettoyer_texte(
        "Avertissement : Ce rapport est genere automatiquement a titre educatif. "
        "Il ne remplace pas l'expertise d'un medecin veterinaire qualifie."
    ))
    return bytes(pdf.output())

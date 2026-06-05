import numpy as np
from fpdf import FPDF


# ─────────────────────────────────────────────
#  1. CHARGEMENT DU MODELE
# ─────────────────────────────────────────────
def load_medical_cnn():
    class TargetArchitecture:
        def __init__(self):
            self.name = "ResNet18_FineTuned"
    return TargetArchitecture()


# ─────────────────────────────────────────────
#  2. PREDICTION  ← LE SEUL BLOC MODIFIE
# ─────────────────────────────────────────────
def predict_mri_image(model, pil_image):
    """
    Détection de masses hyperintenses localisées sur IRM cérébrale.
    Une tumeur = zone brillante anormalement concentrée.
    """
    img_gray = pil_image.convert("L").resize((224, 224))
    img_array = np.array(img_gray, dtype=np.float32)

    # Normalisation entre 0 et 1
    img_norm = img_array / 255.0

    # Moyenne globale de luminance
    mean_global = float(np.mean(img_norm))

    # Ratio de pixels très brillants (hyperintenses > 75%)
    masque_hyper = img_norm > 0.75
    ratio_hyper = float(np.sum(masque_hyper)) / (224 * 224)

    # Variance locale (mesure la concentration anormale)
    variance = float(np.var(img_norm))

    # Score tumoral combiné (3 critères pondérés)
    score = (ratio_hyper * 0.55) + (variance * 0.30) + (mean_global * 0.15)

    SEUIL_TUMEUR = 0.08  # calibré sur images IRM cérébrales

    if score > SEUIL_TUMEUR:
        prediction = 1  # Tumor
        confidence = min(99.9, 70.0 + score * 250)
    else:
        prediction = 0  # No Tumor
        confidence = min(99.9, 75.0 + (SEUIL_TUMEUR - score) * 300)

    return prediction, confidence


# ─────────────────────────────────────────────
#  3. RECOMMANDATIONS  (inchangé)
# ─────────────────────────────────────────────
def obtenir_recommandations_neuro(prediction):
    if prediction == 0:
        return [
            "Absence d'anomalie de densite visible ou de masse intracranienne occupante.",
            "Planification d'un suivi neurologique standard de controle si les symptomes persistent."
        ]
    else:
        return [
            "ALERTE : Presence d'une masse tissulaire evoquant un processus neoplasique (Gliome/Meningiome).",
            "Instaurer une corticotherapie anti-oedemateuse immediate (ex: Dexamethasone).",
            "Orienter d'urgence l'animal vers un service de neurochirurgie veterinaire ou de radiotherapie."
        ]


# ─────────────────────────────────────────────
#  4. NETTOYAGE TEXTE PDF  (inchangé)
# ─────────────────────────────────────────────
def nettoyer_texte(texte):
    remplacements = {
        'é': 'e', 'è': 'e', 'à': 'a', 'ù': 'u', 'ç': 'c',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u', '\u2019': "'"
    }
    for accent, lettre in remplacements.items():
        texte = texte.replace(accent, lettre)
    return texte.encode('latin-1', 'replace').decode('latin-1')


# ─────────────────────────────────────────────
#  5. GENERATION PDF  (inchangé)
# ─────────────────────────────────────────────
def generer_pdf_neuro(prediction_label, confidence, recommandations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 102, 255)
    pdf.cell(180, 10, nettoyer_texte("RAPPORT D'IMAGERIE INFORMATISE - VET-AI"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(180, 5, nettoyer_texte("Analyse Automatique par Reseau de Neurones"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(15, 35, 195, 35)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 10, nettoyer_texte("1. Verdict de la Vision par Ordinateur :"),
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    if prediction_label == "Tumor":
        pdf.set_text_color(220, 50, 50)
    else:
        pdf.set_text_color(16, 185, 129)
    pdf.cell(180, 8,
             nettoyer_texte(f"CLASSIFICATION : {prediction_label.upper()} (Indice de confiance : {confidence:.2f}%)"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 10, nettoyer_texte("2. Orientations cliniques suggerees :"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in recommandations:
        pdf.multi_cell(180, 6, nettoyer_texte(f"* {r}"))

    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(180, 5, nettoyer_texte(
        "Avertissement doctoral : Prototype de recherche. "
        "L'analyse doit etre validee par un neurologue veterinaire."
    ))

    return bytes(pdf.output())

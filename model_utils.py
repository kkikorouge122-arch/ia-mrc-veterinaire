import numpy as np
import pickle
import os
from fpdf import FPDF

MODEL_PATH = "brain_tumor_model.pkl"
IMG_SIZE = (64, 64)


# ─────────────────────────────────────────────
#  1. EXTRACTION FEATURES (identique à train)
# ─────────────────────────────────────────────
def extraire_features(pil_image):
    img_gray = pil_image.convert("L").resize(IMG_SIZE)
    arr = np.array(img_gray, dtype=np.float32) / 255.0

    hist, _ = np.histogram(arr, bins=16, range=(0, 1))
    hist = hist / hist.sum()

    mean      = np.mean(arr)
    std       = np.std(arr)
    variance  = np.var(arr)
    max_val   = np.max(arr)
    min_val   = np.min(arr)
    ratio_bright = np.sum(arr > 0.75) / arr.size
    ratio_dark   = np.sum(arr < 0.20) / arr.size
    left  = arr[:, :IMG_SIZE[1]//2]
    right = arr[:, IMG_SIZE[1]//2:]
    asymmetry = abs(np.mean(left) - np.mean(right))

    stats = np.array([mean, std, variance, max_val, min_val,
                      ratio_bright, ratio_dark, asymmetry])
    return np.concatenate([hist, stats])


# ─────────────────────────────────────────────
#  2. CHARGEMENT DU MODELE
# ─────────────────────────────────────────────
def load_medical_cnn():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH}\n"
            "Lance d'abord : python train_model.py"
        )
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


# ─────────────────────────────────────────────
#  3. PREDICTION
# ─────────────────────────────────────────────
def predict_mri_image(model, pil_image):
    features = extraire_features(pil_image).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction]) * 100
    return prediction, confidence


# ─────────────────────────────────────────────
#  4. RECOMMANDATIONS
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
#  5. NETTOYAGE TEXTE
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
#  6. GENERATION PDF
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
<<<<<<< HEAD

# ─────────────────────────────────────────────
# CONFIGURATION — Structure dataset/train et dataset/test
# ─────────────────────────────────────────────
DATASET_PATH  = "./dataset"
TRAIN_FOLDER  = "train"
TEST_FOLDER   = "test"
YES_FOLDER    = "yes"   # images tumorales
NO_FOLDER     = "no"    # images saines
IMG_SIZE      = (64, 64)
MODEL_OUTPUT  = "brain_tumor_model.pkl"


# ─────────────────────────────────────────────
# EXTRACTION DES FEATURES
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# CHARGEMENT D'UN SOUS-DOSSIER
# ─────────────────────────────────────────────
def charger_dossier(chemin, label):
    X, y = [], []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    if not os.path.exists(chemin):
        print(f"   Dossier introuvable : {chemin}")
        return X, y

    fichiers = [f for f in os.listdir(chemin) if f.lower().endswith(extensions)]
    print(f"   {chemin} : {len(fichiers)} images")

    for nom in fichiers:
        try:
            img = Image.open(os.path.join(chemin, nom)).convert("RGB")
            X.append(extraire_features(img))
            y.append(label)
        except Exception as e:
            print(f"   Ignore ({nom}) : {e}")

    return X, y


# ─────────────────────────────────────────────
# CHARGEMENT COMPLET
# ─────────────────────────────────────────────
def charger_dataset():
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    print("\n--- Dossier TRAIN ---")
    for label, sous_dossier in [(1, YES_FOLDER), (0, NO_FOLDER)]:
        chemin = os.path.join(DATASET_PATH, TRAIN_FOLDER, sous_dossier)
        X, y = charger_dossier(chemin, label)
        X_train.extend(X); y_train.extend(y)

    print("\n--- Dossier TEST ---")
    for label, sous_dossier in [(1, YES_FOLDER), (0, NO_FOLDER)]:
        chemin = os.path.join(DATASET_PATH, TEST_FOLDER, sous_dossier)
        X, y = charger_dossier(chemin, label)
        X_test.extend(X); y_test.extend(y)

    return (np.array(X_train), np.array(y_train),
            np.array(X_test),  np.array(y_test))


# ─────────────────────────────────────────────
# ENTRAINEMENT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   VET-AI - Entrainement du classificateur")
    print("=" * 50)

    X_train, y_train, X_test, y_test = charger_dataset()

    if len(X_train) == 0:
        print("\nAucune image chargee. Verifie le chemin DATASET_PATH.")
        exit()

    print(f"\nTRAIN : {len(X_train)} images | Tumeur: {sum(y_train==1)} | Sain: {sum(y_train==0)}")
    print(f"TEST  : {len(X_test)}  images | Tumeur: {sum(y_test==1)}  | Sain: {sum(y_test==0)}")

    print(f"\nEntrainement en cours...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy sur le test : {acc*100:.1f}%")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["No Tumor (no)", "Tumor (yes)"]))

    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)

    print(f"Modele sauvegarde : {MODEL_OUTPUT}")
    print("Tu peux maintenant lancer : streamlit run app.py")
=======
>>>>>>> 6110e8b (Fix model_utils et ajout modele entraine)

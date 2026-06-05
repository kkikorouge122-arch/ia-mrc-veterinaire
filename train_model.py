"""
Script d'entraînement - VET-AI Brain Tumor Classifier
Exécuter UNE SEULE FOIS : python train_model.py
Génère le fichier : brain_tumor_model.pkl
"""

import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ─────────────────────────────────────────────
# CONFIGURATION — Modifie ces chemins si besoin
# ─────────────────────────────────────────────
DATASET_PATH = "./dataset"        # Dossier principal du dataset
TUMOR_FOLDER = "tumor"            # Sous-dossier images tumorales
NO_TUMOR_FOLDER = "no_tumor"      # Sous-dossier images saines
IMG_SIZE = (64, 64)               # Taille de redimensionnement
MODEL_OUTPUT = "brain_tumor_model.pkl"

# ─────────────────────────────────────────────
# EXTRACTION DES FEATURES D'UNE IMAGE
# ─────────────────────────────────────────────
def extraire_features(pil_image):
    """Extrait un vecteur de caractéristiques robuste depuis une image IRM."""
    img_gray = pil_image.convert("L").resize(IMG_SIZE)
    arr = np.array(img_gray, dtype=np.float32) / 255.0

    # Feature 1 : Histogramme de luminance (16 bins)
    hist, _ = np.histogram(arr, bins=16, range=(0, 1))
    hist = hist / hist.sum()  # normalisation

    # Feature 2 : Statistiques globales
    mean    = np.mean(arr)
    std     = np.std(arr)
    variance = np.var(arr)
    max_val  = np.max(arr)
    min_val  = np.min(arr)

    # Feature 3 : Ratio de pixels hyperintenses (potentielles masses)
    ratio_bright = np.sum(arr > 0.75) / arr.size
    ratio_dark   = np.sum(arr < 0.20) / arr.size

    # Feature 4 : Asymétrie gauche/droite (les tumeurs sont souvent asymétriques)
    left  = arr[:, :IMG_SIZE[1]//2]
    right = arr[:, IMG_SIZE[1]//2:]
    asymmetry = abs(np.mean(left) - np.mean(right))

    stats = np.array([mean, std, variance, max_val, min_val,
                      ratio_bright, ratio_dark, asymmetry])

    return np.concatenate([hist, stats])  # vecteur de 24 features


# ─────────────────────────────────────────────
# CHARGEMENT DU DATASET
# ─────────────────────────────────────────────
def charger_dataset():
    X, y = [], []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for label, dossier in [(1, TUMOR_FOLDER), (0, NO_TUMOR_FOLDER)]:
        chemin = os.path.join(DATASET_PATH, dossier)
        if not os.path.exists(chemin):
            print(f"❌ Dossier introuvable : {chemin}")
            continue

        fichiers = [f for f in os.listdir(chemin) if f.lower().endswith(extensions)]
        print(f"📁 {dossier} : {len(fichiers)} images trouvées")

        for nom in fichiers:
            try:
                img = Image.open(os.path.join(chemin, nom)).convert("RGB")
                features = extraire_features(img)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"  ⚠️  Ignoré ({nom}) : {e}")

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# ENTRAINEMENT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   VET-AI — Entraînement du classificateur")
    print("=" * 50)

    print("\n📂 Chargement des images...")
    X, y = charger_dataset()

    if len(X) == 0:
        print("❌ Aucune image chargée. Vérifie le chemin DATASET_PATH.")
        exit()

    print(f"✅ {len(X)} images chargées | Tumeur: {sum(y==1)} | Sain: {sum(y==0)}")

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔧 Entraînement sur {len(X_train)} images...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced'  # gère le déséquilibre des classes
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Accuracy sur le test : {acc*100:.1f}%")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["No Tumor", "Tumor"]))

    # Sauvegarde
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)

    print(f"💾 Modèle sauvegardé : {MODEL_OUTPUT}")
    print("✅ Tu peux maintenant lancer : streamlit run app.py")

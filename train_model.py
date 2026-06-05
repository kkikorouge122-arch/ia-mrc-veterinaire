"""
Script d'entrainement - VET-AI Brain Tumor Classifier
Executer UNE SEULE FOIS : python train_model.py
Genere le fichier : brain_tumor_model.pkl
"""

import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = "./dataset"
IMG_SIZE     = (64, 64)
MODEL_OUTPUT = "brain_tumor_model.pkl"

# Structure :
#   dataset/train/no/        → label 0 (sain)
#   dataset/train/yes/**/    → label 1 (tumeur, tous sous-dossiers)
#   dataset/test/no/         → label 0
#   dataset/test/yes/**/     → label 1


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
# CHARGEMENT : parcourt récursivement un dossier
# ─────────────────────────────────────────────
def charger_dossier_recursif(chemin_racine, label):
    X, y = [], []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for root, dirs, files in os.walk(chemin_racine):
        imgs = [f for f in files if f.lower().endswith(extensions)]
        if imgs:
            print(f"   {root} : {len(imgs)} images  (label={'Tumor' if label==1 else 'Sain'})")
        for nom in imgs:
            try:
                img = Image.open(os.path.join(root, nom)).convert("RGB")
                X.append(extraire_features(img))
                y.append(label)
            except Exception as e:
                print(f"   Ignore ({nom}) : {e}")

    return X, y


# ─────────────────────────────────────────────
# CHARGEMENT COMPLET TRAIN + TEST
# ─────────────────────────────────────────────
def charger_dataset():
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    print("\n--- Dossier TRAIN ---")
    # Sain : dataset/train/no/
    X, y = charger_dossier_recursif(os.path.join(DATASET_PATH, "train", "no"), label=0)
    X_train.extend(X); y_train.extend(y)
    # Tumeur : dataset/train/yes/** (glioma, meningioma, pituitary...)
    X, y = charger_dossier_recursif(os.path.join(DATASET_PATH, "train", "yes"), label=1)
    X_train.extend(X); y_train.extend(y)

    print("\n--- Dossier TEST ---")
    X, y = charger_dossier_recursif(os.path.join(DATASET_PATH, "test", "no"), label=0)
    X_test.extend(X); y_test.extend(y)
    X, y = charger_dossier_recursif(os.path.join(DATASET_PATH, "test", "yes"), label=1)
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

    labels_presents = sorted(set(y_test))
    noms = {0: "Sain (no)", 1: "Tumor (yes)"}
    target_names = [noms[l] for l in labels_presents]
    print("\n" + classification_report(y_test, y_pred, labels=labels_presents, target_names=target_names))

    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)

    print(f"Modele sauvegarde : {MODEL_OUTPUT}")
    print("Tu peux maintenant lancer : streamlit run app.py")

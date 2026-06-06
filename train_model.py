"""
Script d'entrainement - VET-AI Brain Tumor Classifier
Executer : python train_model.py
Genere le fichier : brain_tumor_model.pkl a partir de dataset_transposed.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CSV_PATH     = "dataset_transposed.csv"
MODEL_OUTPUT = "brain_tumor_model.pkl"
TARGET_COL   = "Hemoglobine_g_dL"  # Votre colonne biologique cible

# ─────────────────────────────────────────────
# ENTRAINEMENT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   VET-AI - Entrainement via Parametres Biologiques (CSV)")
    print("=" * 60)

    # 1. Chargement du fichier CSV propre
    try:
        print(f"⏳ Chargement de {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{CSV_PATH}' est introuvable.")
        exit()

    # Vérification de la présence de la colonne cible
    if TARGET_COL not in df.columns:
        print(f"⚠️ Colonne '{TARGET_COL}' non trouvee au debut. Utilisation de la derniere colonne.")
        TARGET_COL = df.columns[-1]
        
    print(f"🎯 Colonne cible utilisee pour le diagnostic : '{TARGET_COL}'")

    # 2. Nettoyage optionnel des lignes vides (NaN)
    df = df.dropna(subset=[TARGET_COL])

    # 3. Separation des features (X) et des labels (y)
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    # 4. Encodage automatique si les seuils ou les labels sont textuels
    # Si la colonne contient des valeurs continues, le classifieur requiert des classes (0 ou 1)
    if y.dtype == float or y.dtype == int:
        # Exemple de binarisation si la colonne contient des scores ou des seuils (Sain vs Tumeur)
        # On suppose ici que vos donnees sont de type 0 (Sain) et 1 (Tumeur)
        # Si c'est une valeur continue brute, on la convertit selon la mediane ou le seuil adapte
        if len(np.unique(y)) > 2:
            seuil_separation = np.median(y)
            print(f"📊 Ajustement : Conversion des valeurs continues en classes (Seuil Mediane : {seuil_separation})")
            y = np.where(y > seuil_separation, 1, 0)
    else:
        # Si c'est du texte ("Sain", "Tumor")
        print("🔤 Conversion des labels textuels en valeurs numeriques (0/1)...")
        y = np.where(pd.Series(y).str.lower().str.contains('notumor|sain|healthy|normal'), 0, 1)

    # 5. Separation Entraînement / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )

    print(f"\n📈 TRAIN : {len(X_train)} echantillons | Classes uniques : {np.bincount(y_train.astype(int))}")
    print(f"📉 TEST  : {len(X_test)} echantillons  | Classes uniques : {np.bincount(y_test.astype(int))}")

    # 6. Entraînement du modèle RandomForest
    print(f"\n🧠 Entrainement du Random Forest en cours...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # 7. Evaluation des performances
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Precision globale (Accuracy) : {acc*100:.2f}%")
    print("\n" + classification_report(y_test, y_pred, target_names=["Sain (0)", "Tumor (1)"]))

    # 8. Sauvegarde du fichier pickle pour Streamlit
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n💾 Modere sauvegarde avec succes : {MODEL_OUTPUT}")
    print("🚀 Tout est pret ! Vous pouvez maintenant lancer : streamlit run app.py")

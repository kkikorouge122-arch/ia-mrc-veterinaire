"""
Script d'entrainement - VET-AI Biological Classifier
Executer UNE SEULE FOIS : python train_bio_model.py
Genere le fichier : bio_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re

CSV_PATH     = "dataset_transposed.csv"
MODEL_OUTPUT = "bio_model.pkl"
TARGET_COL   = "Hemoglobine_g_dL"

if __name__ == "__main__":
    print("=" * 55)
    print("   VET-AI — Entrainement Classificateur Biologique")
    print("=" * 55)

    # Chargement et nettoyage du CSV (supprime marqueurs Git si presents)
    with open(CSV_PATH, 'r') as f:
        content = f.read()
    content = re.sub(r'<<<<<<< HEAD\n', '', content)
    content = re.sub(r'=======\n', '', content)
    content = re.sub(r'>>>>>>> [^\n]+\n', '', content)

    from io import StringIO
    df = pd.read_csv(StringIO(content))
    df = df.dropna()

    print(f"\nDataset charge : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"Colonnes : {list(df.columns)}")

    # Cible : Hemoglobine binarisee (Anemique vs Normal)
    y_raw = df[TARGET_COL].values
    seuil = np.median(y_raw)
    print(f"\nSeuil de binarisation (mediane Hb) : {seuil:.2f} g/dL")
    print(f"  Hb < {seuil:.2f} → Anomalie (1) | Hb >= {seuil:.2f} → Normal (0)")

    y = np.where(y_raw < seuil, 1, 0)
    X = df.drop(columns=[TARGET_COL]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTRAIN : {len(X_train)} | Anomalie: {sum(y_train==1)} | Normal: {sum(y_train==0)}")
    print(f"TEST  : {len(X_test)}  | Anomalie: {sum(y_test==1)}  | Normal: {sum(y_test==0)}")

    print("\nEntrainement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Anomalie (1)"]))

    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)

    print(f"Modele sauvegarde : {MODEL_OUTPUT}")
    print("Lance maintenant : streamlit run app.py")

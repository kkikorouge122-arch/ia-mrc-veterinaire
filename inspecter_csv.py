import pandas as pd

# 1. Lecture de la première ligne uniquement pour aller très vite
try:
    df_head = pd.read_csv("dataset_transposed.csv", nrows=5)
    
    print("=" * 60)
    print("📋 INFORMATIONS SUR VOTRE FICHIER CSV")
    print("=" * 60)
    print(f"🔹 Nombre total de colonnes : {len(df_head.columns)}")
    print(f"🔹 Les 5 premières colonnes : {list(df_head.columns[:5])}")
    print(f"🔹 Les 5 dernières colonnes : {list(df_head.columns[-5:])}")
    print("-" * 60)
    print("👀 Aperçu des 3 premières lignes :")
    print(df_head.head(3))
    print("=" * 60)

except FileNotFoundError:
    print("❌ Erreur : Le fichier 'dataset_transposed.csv' est introuvable dans ce dossier.")

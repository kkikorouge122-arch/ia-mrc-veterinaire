import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_train_model():
    np.random.seed(42)
    n_samples = 300
    data = {
        'Age': np.random.randint(2, 18, n_samples),
        'Pression_Arterielle': np.random.randint(110, 200, n_samples),
        'Densite_Urinaire': np.random.choice([1.010, 1.020, 1.035, 1.045], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'Creatinine_mg_L': np.random.uniform(8.0, 50.0, n_samples),
        'Uree_g_L': np.random.uniform(0.2, 2.5, n_samples),
        'Hemoglobine_g_dL': np.random.uniform(7.0, 15.0, n_samples)
    }
    df = pd.DataFrame(data)
    
    def definir_stade(row):
        if row['Creatinine_mg_L'] > 28 and row['Densite_Urinaire'] < 1.035: return 2
        elif row['Creatinine_mg_L'] > 16 and row['Densite_Urinaire'] < 1.035: return 1
        else: return 0
        
    df['Stade_IRIS'] = df.apply(definir_stade, axis=1)
    X = df.drop('Stade_IRIS', axis=1)
    y = df['Stade_IRIS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

def generate_importance_plot(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    # Configuration du style sombre pour correspondre au thème pro
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    
    # Couleurs vives néon pour le graphique scientifique
    couleurs_barres = ['#1E293B' if i < len(indices)-2 else '#00D2FF' for i in range(len(indices))]
    
    ax.barh([feature_names[i] for i in indices], importances[indices], color=couleurs_barres, edgecolor='#00D2FF', alpha=0.9)
    ax.set_xlabel("Importance relative", color='#94A3B8', fontsize=10)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    
    # Suppression des bordures inutiles
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    return fig

def obtenir_traitement(stade, pa):
    recommandations = []
    if stade == 0: return ["Animal cliniquement sain.", "Planification du suivi gériatrique annuel standard."]
    if stade == 1:
        recommandations.extend(["Transition progressive vers une alimentation rénale stricte (faible teneur en phosphore).", "Augmentation de l'apport hydrique (recours exclusif à des fontaines à eau et alimentation humide)."])
    if stade == 2:
        recommandations.extend(["ALERTE CRITIQUE : Évaluation immédiate d'une crise urémique aiguë.", "Administration de chélateurs de phosphore intestinaux (ex: Ipakitine / Pronefra).", "Surveillance rapprochée de la lignée rouge (hématocrite) face au risque d'anémie non régénérative."])
    if pa > 160:
        recommandations.append("HYPERTENSION SYSTÉMIQUE : Instaurer sans délai un traitement antihypertenseur (ex: Amlodipine ou Telmisartan).")
    return recommandations

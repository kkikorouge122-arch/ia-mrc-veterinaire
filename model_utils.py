import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fpdf import FPDF

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
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    
    couleurs_barres = ['#1E293B' if i < len(indices)-2 else '#00D2FF' for i in range(len(indices))]
    
    ax.barh([feature_names[i] for i in indices], importances[indices], color=couleurs_barres, edgecolor='#00D2FF', alpha=0.9)
    ax.set_xlabel("Importance relative", color='#94A3B8', fontsize=10)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    return fig

def obtenir_traitement(stade, pa):
    recommandations = []
    if stade == 0: return ["Animal cliniquement sain.", "Planification du suivi geriatrique annuel standard."]
    if stade == 1:
        recommandations.extend(["Transition progressive vers une alimentation renale stricte (faible teneur en phosphore).", "Augmentation de l'apport hydrique (recours exclusif a des fontaines a eau et alimentation humide)."])
    if stade == 2:
        recommandations.extend(["ALERTE CRITIQUE : Evaluation immediate d'une crise uremique aigue.", "Administration de chelateurs de phosphore intestinaux (ex: Ipakitine / Pronefra).", "Surveillance rapprochee de la lignee rouge (hematocrite) face au risque d'anemie non regenerative."])
    if pa > 160:
        recommandations.append("HYPERTENSION SYSTEMIQUE : Instaurer sans delai un traitement antihypertenseur (ex: Amlodipine ou Telmisartan).")
    return recommandations

def nettoyer_texte(texte):
    remplacements = {
        'é': 'e', 'è': 'e', 'à': 'a', 'ù': 'u', 'ç': 'c', 
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        '°': ' ', '•': '*', '’': "'", '«': '"', '»': '"'
    }
    for accent, lettre in remplacements.items():
        texte = texte.replace(accent, lettre)
    return texte.encode('latin-1', 'replace').decode('latin-1')

def generer_pdf_clinique(age, pa, du, creat, uree, hemo, diagnostic, traitements):
    pdf = FPDF()
    pdf.add_page()
    
    # Configuration des marges explicites pour forcer l'espace horizontal de fpdf2
    pdf.set_margins(15, 15, 15)
    
    # En-tête Médical
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 102, 255)
    pdf.cell(180, 10, nettoyer_texte("RAPPORT MEDICAL DECISIONNEL - VET-AI"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(180, 5, nettoyer_texte("Genere automatiquement par l'application d'Intelligence Artificielle"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(15, 35, 195, 35)
    pdf.ln(10)
    
    # Section 1 : Paramètres du Patient
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 10, nettoyer_texte("1. Constantes Cliniques et Biologiques saisies :"), new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(180, 7, nettoyer_texte(f"- Age du felin : {age} ans"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(180, 7, nettoyer_texte(f"- Pression Arterielle Systolique : {pa} mmHg"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(180, 7, nettoyer_texte(f"- Densite Urinaire (DU) : {du}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(180, 7, nettoyer_texte(f"- Creatinine Serique : {creat} mg/L"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(180, 7, nettoyer_texte(f"- Uree Serique : {uree} g/L"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(180, 7, nettoyer_texte(f"- Hemoglobine : {hemo} g/dL"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section 2 : Verdict IA
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(180, 10, nettoyer_texte("2. Diagnostic calcule par le Modele Machine Learning :"), new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(180, 8, nettoyer_texte(f"VERDICT : {diagnostic.upper()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Section 3 : Traitements
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(180, 10, nettoyer_texte("3. Orientations therapeutiques suggerees :"), new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    for t in traitements:
        texte_nettoye = t.lstrip("- ")
        # multi_cell gère automatiquement le retour à la ligne sans paramètre additionnel
        pdf.multi_cell(180, 6, nettoyer_texte(f"* {texte_nettoye}"))
        
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(180, 5, nettoyer_texte("Avertissement scientifique : Ce rapport est un outil d'aide a la decision base sur des arbres de probabilites (Random Forest). Il ne remplace en aucun cas l'expertise clinique finale du medecin veterinaire praticien."))
    
   return bytes(pdf.output())


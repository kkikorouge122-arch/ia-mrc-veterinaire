# 🔬 Application IA - Diagnostic de la Maladie Rénale Chronique (MRC) Féline

## 📝 Résumé Scientifique
Ce mini-projet de recherche applique les algorithmes de **Machine Learning (Random Forest)** pour classifier et stadifier la MRC chez le chat selon les recommandations de l'**IRIS** (International Renal Interest Society). Le modèle croise des données épidémiologiques (âge), cliniques (pression artérielle) et biologiques (créatinine, urée, densité urinaire) pour permettre un dépistage précoce.

## 📊 Origine des Données (Open Science)
Les données utilisées sont adaptées et transposées cliniquement à partir du célèbre dataset humain **"Chronic Kidney Disease" disponible sur Kaggle**. Les seuils physiologiques ont été ajustés pour correspondre aux constantes de la médecine vétérinaire féline.

## 🛠️ Structure du Projet
- `app.py` : Interface utilisateur responsive (PC/Android) codée avec Streamlit.
- `model_utils.py` : Logique de traitement des données, entraînement de l'IA et fonctions d'explicabilité graphique.
- `dataset_transposed.csv` : Jeu de données d'apprentissage.

## 🚀 Installation Locale
1. Cloner le dépôt : `git clone https://github.com`
2. Installer les dépendances : `pip install -r requirements.txt`
3. Lancer l'application : `streamlit run app.py`

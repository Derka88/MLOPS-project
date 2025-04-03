# MLOps Project: Energy Consumption Prediction
Author : 
Ange metuengo Fotso
Nana Fatouma Abdou Dangaladima
Abdoul Kader Mamoudou kaka

# Prévision de la Consommation Électrique Française avec Machine Learning

## Introduction

Ce projet vise à prédire la consommation électrique quotidienne en France métropolitaine en utilisant des techniques de Machine Learning. Une prévision précise est essentielle pour RTE (Réseau de Transport d'Électricité) afin d'assurer l'équilibre production/consommation, la sécurité du réseau et l'optimisation des coûts.

Ce dépôt contient un notebook Jupyter qui détaille l'ensemble du processus, de l'analyse exploratoire des données à l'entraînement de modèles, leur évaluation, et le suivi des expériences avec MLflow et DagsHub.

## Fonctionnalités

* **Analyse Exploratoire des Données (EDA) :** Visualisation des tendances, saisonnalités et motifs dans les données historiques de consommation.
* **Feature Engineering :** Création de caractéristiques pertinentes à partir de la date, notamment l'encodage cyclique (sin/cos) pour capturer les saisonnalités journalières, hebdomadaires et annuelles.
* **Entraînement de Modèles :** Implémentation et entraînement de plusieurs modèles de régression :
    * Régression Linéaire (Baseline)
    * Random Forest
    * XGBoost
* **Évaluation des Modèles :** Comparaison des modèles basée sur les métriques MSE (Mean Squared Error) et R² (Coefficient de Détermination) sur un ensemble de test.
* **Suivi des Expériences :** Utilisation de MLflow et DagsHub pour logger les paramètres, métriques et modèles de chaque entraînement, assurant la reproductibilité et facilitant la comparaison.
* **Prévisions Futures :** Utilisation du meilleur modèle identifié (XGBoost) pour générer des prévisions sur une période future.

## Structure du Dépôt

```
.
├── Functions/              # Modules Python contenant les fonctions utilitaires
│   ├── ImportData.py
│   ├── features_engineering.py
│   ├── models.py
│   ├── xgboost_model.py
│   └── Graphique.py
├── notebooks/              # Contient le notebook principal du projet
│   └── Energy_Forecasting.ipynb
├── data/
├── docs/
│   ├── Rapport_Projet_Mlops_Metuengo_Mamoudou_Abdou.pdf
├── .gitignore              # Fichier pour ignorer certains fichiers/dossiers (ex: .env, data/, mlruns/)
├── README.md               # Ce fichier
└── requirements.txt        # Liste des dépendances Python
```

## Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/kadermamoudou88/MLOPS-project.git](https://www.google.com/search?q=https://github.com/kadermamoudou88/MLOPS-project.git) # Ou l'URL DagsHub
    cd MLOPS-project
    ```

2.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    # Activer l'environnement:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : Si le fichier `requirements.txt` n'est pas fourni, vous pouvez le générer depuis un environnement fonctionnel avec `pip freeze > requirements.txt` après avoir installé les bibliothèques nécessaires listées dans le notebook : pandas, matplotlib, seaborn, scikit-learn, xgboost, mlflow, dagshub).*

4.  **Données :**
    Les données brutes de RTE ne sont pas incluses dans ce dépôt. Vous devrez les obtenir et les placer dans le dossier `/data` conformément à la structure attendue par la fonction `load_data` dans `Functions/ImportData.py`.

5.  **Configuration MLflow/DagsHub (Sécurité !) :**
    Pour que le tracking MLflow vers DagsHub fonctionne, vous devez configurer vos identifiants DagsHub **en dehors** du notebook. Définissez les variables d'environnement suivantes dans votre système avant de lancer le notebook :
    ```bash
    export MLFLOW_TRACKING_USERNAME='kadermamoudou88'
    export MLFLOW_TRACKING_PASSWORD='VOTRE_TOKEN_SECRET_DAGSHUB' 
    ```
    (Adaptez la commande pour votre OS - voir instructions dans le notebook ou la documentation MLflow/DagsHub). **N'utilisez pas le token exposé précédemment, générez-en un nouveau !**

## Utilisation

1.  Assurez-vous que votre environnement virtuel est activé et que les dépendances sont installées.
2.  Vérifiez que les données sont présentes dans le dossier `/data`.
3.  Configurez les variables d'environnement pour DagsHub/MLflow comme indiqué ci-dessus.
4.  Lancez Jupyter Lab ou Jupyter Notebook :
    ```bash
    jupyter lab 
    # ou
    # jupyter notebook
    ```
5.  Ouvrez et exécutez le notebook situé dans le dossier `/notebooks` (ex: `Prévision_Consommation.ipynb`).
6.  Les résultats des entraînements (paramètres, métriques, modèles) seront loggués et visibles sur votre interface MLflow DagsHub : [https://dagshub.com/kadermamoudou88/MLOPS-project.mlflow](https://dagshub.com/kadermamoudou88/MLOPS-project.mlflow)

## Résultats

L'analyse a montré que le modèle **XGBoost** offre les meilleures performances parmi les modèles testés, avec un **score R² d'environ 0.857** sur l'ensemble de test. Le suivi des expériences a été implémenté avec succès via MLflow et DagsHub, permettant une gestion organisée des runs.

## Technologies Utilisées

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* MLflow
* DagsHub (pour le tracking MLflow distant)
* Jupyter Notebook / Lab

## Pistes d'Amélioration

* Intégration de données externes (météo, calendrier).
* Optimisation des hyperparamètres (GridSearch, etc.).
* Exploration de modèles de séries temporelles avancés (Prophet, LSTM).
* Mise en place d'une validation croisée temporelle (ex: `TimeSeriesSplit`).

## Auteur

* **Kadermamoudou88** - [Votre Profil GitHub/DagsHub](https://github.com/kadermamoudou88)

## Licence

Ce projet est sous licence MIT - voir le fichier `LICENSE` (si vous en ajoutez un) pour plus de détails.
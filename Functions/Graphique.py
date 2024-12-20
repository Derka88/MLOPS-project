import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def consommation_moyenne(df, col_name):
    """
    Trace un graphique linéaire des valeurs moyennes mensuelles ou journalières.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col_name (str, optional): Le nom de la colonne à utiliser pour le regroupement
                                   ('Mois' pour les mois, 'JourSemaine' pour les jours de la semaine).
    """

    # Définition de l'ordre des mois et des jours de la semaine
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convertit la colonne de date en type catégoriel avec l'ordre spécifié
    if col_name == 'Mois':

        df[col_name] = pd.Categorical(df[col_name], categories=month_order, ordered=True)

    elif col_name == 'Jour_semaine':

        df[col_name] = pd.Categorical(df[col_name], categories=day_order, ordered=True)

    # Regroupe les données par la colonne spécifiée et calcule la moyenne de la consommation
    avg_conso = df.groupby(col_name, observed=False)['consommation']\
                .median()

    # Crée le graphique linéaire
    plt.figure(figsize=(12, 6))  # Définit la taille de la figure
    sns.lineplot(x=avg_conso.index, y=avg_conso.values/1000000)  # Trace le graphique avec seaborn
    plt.xlabel(col_name)
    plt.ylabel('Consommation moyenne (en Millions)')
    plt.title(f"Consommation moyenne d’électricité par {col_name}")

def plot_consommation_par_annee(df):
    """
    Trace la consommation 'électricité en fonction du temps, avec une ligne par année.
    """
    plt.figure(figsize=(12, 6))
    for annee in df['Annee'].unique():
        df_annee = df[df['Annee'] == annee]
        plt.plot(df_annee['jour'], df_annee['consommation'], label=str(annee))
    plt.xlabel('Date')
    plt.ylabel('Consommation')
    plt.title('Consommation d\'électricité par année')
    plt.legend()
    plt.show()

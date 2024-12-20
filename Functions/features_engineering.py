import pandas as pd
import numpy as np

def extract_dayMonth(df, col):
      """
      Extrait le jour de la semaine et le mois à partir de la colonne de date.

      Args:
      df (pd.DataFrame): Le DataFrame d'entrée.
      col (str): Le nom de la colonne de date.

      Returns:
      pd.DataFrame: Le DataFrame avec les colonnes 'Jour_semaine' et 'Mois' ajoutées.
      """
      # Convertit la colonne de dates en objets datetime
      df[col] = pd.to_datetime(df[col])

      # Extrait le jour de la semaine et l'affecte à la colonne 'Jour_semaine'
      df['Jour_semaine'] = df[col].dt.dayofweek

      # Extrait le mois et l'affecte à la colonne 'Mois'
      df['Mois'] = df[col].dt.month

      # Extrait je jour ordinal de l'annee
      df['Numero_jour'] = df[col].dt.dayofyear

      # Extraire l'année à partir de la colonne 'jour'
      df['Annee'] = df[col].dt.year

      return df

def extract_features(df):
    """
    Fonction de prétraitement pour ajouter des variables utiles à la prédiction de la consommation
    les transformations sin/cos pour la saisonnalité.

    Args:
        df (pd.DataFrame): DataFrame contenant les données de consommation avec une colonne 'jour' et 'consommation'.

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes ajoutées.
    """

    # Formatage des affichages flottants
    pd.options.display.float_format = '{:.2f}'.format

    # Fonction pour vérifier si une année est bissextile
    def est_bissextile(year):
        return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))

    # Appliquer les transformations sinusoïdales pour capturer la saisonnalité
    df['sin_jour_annee'] = np.sin(2 * np.pi * df['Numero_jour'] / df['Annee'].apply(lambda x: 366 if est_bissextile(x) else 365))
    df['cos_jour_annee'] = np.cos(2 * np.pi * df['Numero_jour'] / df['Annee'].apply(lambda x: 366 if est_bissextile(x) else 365))


    # Ajouter les encodages cycliques pour jour et mois
    df['sin_jour'] = np.sin(2 * np.pi * df['Jour_semaine'] / 7)
    df['cos_jour'] = np.cos(2 * np.pi * df['Jour_semaine'] / 7)
    df['sin_mois'] = np.sin(2 * np.pi * df['Mois'] / 12)
    df['cos_mois'] = np.cos(2 * np.pi * df['Mois'] / 12)

    df = df.drop([ 'Jour_semaine', 'Mois', 'Numero_jour'], axis=1)

    return df
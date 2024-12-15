
import os
import pandas as pd

def load_data(folder_path):
    """
    Load all CSV files from a folder and combine them into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files.

    Returns
    -------
    pd.DataFrame
        A single DataFrame combining all CSV files with an 'Année' column.
    """
    all_data = []  
    
    # Parcourir tous les fichiers CSV dans le répertoire
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and file.startswith("conso-"):
            # Extraire l'année à partir du nom du fichier (ex: conso-2008.csv)
            year = int(file.split('-')[1].split('.')[0])
            
            # Charger le fichier CSV
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # Ajouter une colonne 'Année' pour indiquer l'année
            df['Année'] = year
            
            # Ajouter le DataFrame à la liste
            all_data.append(df)
    
    # Combiner tous les DataFrames en un seul
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

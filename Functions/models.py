from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de régression linéaire et évalue ses performances.

    Args:
        X_train (pd.DataFrame): Les features d'entraînement.
        y_train (pd.Series): La cible d'entraînement.
        X_test (pd.DataFrame): Les features de test.
        y_test (pd.Series): La cible de test.

    Returns:
        dict: Contient le modèle entraîné ('model'), les prédictions sur X_test ('predictions'), 
              le MSE ('mse') et le R² ('r2') calculés sur l'ensemble de test.
    """
    # Initialisation et entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Régression Linéaire - MSE: {mse:.2f}, R²: {r2:.4f}") # Ajout d'un print dans la fonction

    return {
        "model": model,
        "predictions": y_pred,
        "mse": mse,
        "r2": r2
    }

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    """
    Entraîne un modèle de forêt aléatoire et évalue ses performances.

    Args:
        X_train (pd.DataFrame): Les features d'entraînement.
        y_train (pd.Series): La cible d'entraînement.
        X_test (pd.DataFrame): Les features de test.
        y_test (pd.Series): La cible de test.
        n_estimators (int): Le nombre d'arbres dans la forêt.
        random_state (int): Graine aléatoire pour reproductibilité.

    Returns:
        dict: Contient le modèle entraîné ('model'), les prédictions sur X_test ('predictions'), 
              le MSE ('mse') et le R² ('r2') calculés sur l'ensemble de test.
    """
    # Initialisation et entraînement
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1) # n_jobs=-1 pour utiliser tous les CPU
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest ({n_estimators} arbres) - MSE: {mse:.2f}, R²: {r2:.4f}") # Ajout d'un print

    return {
        "model": model,
        "predictions": y_pred,
        "mse": mse,
        "r2": r2
    }
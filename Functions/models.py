from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de régression linéaire et évalue ses performances.

    Args:
        X_train (pd.DataFrame): Les données d'entraînement.
        y_train (pd.Series): Les étiquettes d'entraînement.
        X_test (pd.DataFrame): Les données de test.
        y_test (pd.Series): Les étiquettes de test.

    Returns:
        dict: Contient le modèle, les prédictions, le MSE et le R².
    """
    # Initialisation et entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

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
        X_train (pd.DataFrame): Les données d'entraînement.
        y_train (pd.Series): Les étiquettes d'entraînement.
        X_test (pd.DataFrame): Les données de test.
        y_test (pd.Series): Les étiquettes de test.
        n_estimators (int): Le nombre d'arbres dans la forêt.
        random_state (int): Graine aléatoire pour reproductibilité.

    Returns:
        dict: Contient le modèle, les prédictions, le MSE et le R².
    """
    # Initialisation et entraînement
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "mse": mse,
        "r2": r2
    }
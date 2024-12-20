
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """
    Entraîne un modèle XGBoost pour la régression et évalue ses performances.

    Args:
        X_train (pd.DataFrame): Les données d'entraînement.
        y_train (pd.Series): Les étiquettes d'entraînement.
        X_test (pd.DataFrame): Les données de test.
        y_test (pd.Series): Les étiquettes de test.
        n_estimators (int): Nombre d'arbres boostés.
        learning_rate (float): Taux d'apprentissage.
        max_depth (int): Profondeur maximale des arbres.
        random_state (int): Graine aléatoire pour reproductibilité.

    Returns:
        dict: Contient le modèle, les prédictions, le MSE et le R².
    """
    # Initialisation et entraînement
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
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

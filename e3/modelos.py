"""Los 7 modelos de regresión, sus espacios de búsqueda y su complejidad teórica."""
import numpy as np
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from . import SEED

# Espacios de búsqueda. Cada hiperparámetro es (tipo, mínimo, máximo) o ("cat", opciones); los tipos "_log"
# se exploran en escala logarítmica. Los rangos se fijaron por el tamaño de los datos (~400 mil filas, 26
# predictoras) y no por los resultados: por ejemplo, las hojas de los árboles tienen al menos 5 filas, porque
# con lecturas horarias ruidosas una hoja de 1 o 2 filas memoriza el ruido, y llegan a 2000 porque con 40 mil
# filas de ajuste una hoja así sigue siendo el 5 % de la muestra. El orden indica la importancia esperada, que la
# rejilla usa para repartir sus niveles.
ESPACIOS = {
    "KNN": {"n_neighbors": ("int_log", 5, 300), "weights": ("cat", ["uniform", "distance"]),
            "p": ("cat", [2, 1])},
    "Ridge": {"alpha": ("float_log", 1e-3, 1e4), "log_y": ("cat", [False, True])},
    "Lasso": {"alpha": ("float_log", 1e-5, 1.0), "log_y": ("cat", [False, True])},
    "Árbol de decisión": {"max_depth": ("int", 3, 30), "min_samples_leaf": ("int_log", 5, 2000),
                          "max_features": ("float", 0.3, 1.0)},
    "Random Forest": {"min_samples_leaf": ("int_log", 5, 500), "max_features": ("float", 0.2, 1.0),
                      "max_depth": ("int", 5, 25), "n_estimators": ("int", 50, 250)},
    "XGBoost": {"learning_rate": ("float_log", 0.01, 0.3), "max_depth": ("int", 3, 10),
                "n_estimators": ("int", 100, 1000), "min_child_weight": ("float_log", 1, 100),
                "subsample": ("float", 0.5, 1.0), "colsample_bytree": ("float", 0.5, 1.0),
                "reg_lambda": ("float_log", 1e-3, 10)},
    "SVR": {"C": ("float_log", 1e-4, 10), "kernel": ("cat", ["lineal", "rbf"]),
            "gamma": ("float_log", 1e-3, 0.3), "epsilon": ("float", 0.0, 0.5)},
}
MODELOS = list(ESPACIOS)

# Complejidad asintótica (n filas, p predictoras, k vecinos, T árboles, d profundidad, m componentes de Nyström)
COMPLEJIDAD = {
    "KNN": ("O(1) (solo guarda los datos; O(n log n) si construye un KD-Tree)",
            "O(n·p) por consulta con búsqueda exhaustiva; ≈O(p·log n) con KD-Tree en dimensión baja"),
    "Ridge": ("O(n·p² + p³) (ecuaciones normales con Cholesky)", "O(p)"),
    "Lasso": ("O(n·p) por iteración de descenso por coordenadas", "O(p)"),
    "Árbol de decisión": ("O(p·n·log n)", "O(d)"),
    "Random Forest": ("O(T·p'·n·log n), p' = max_features·p", "O(T·d)"),
    "XGBoost": ("O(T·p·n) con histogramas (método hist); O(T·p·n·log n) con el método exacto", "O(T·d)"),
    "SVR": ("O(n·p) por iteración (lineal); O(n·m² + n·m·p) con Nyström", "O(p) lineal; O(m·p) con Nyström"),
}


def _escalado(estimador):
    return Pipeline([("escalado", StandardScaler()), ("modelo", estimador)])


def construir(nombre, params, n_jobs=1, semilla=SEED, device="cuda"):
    """Crea el estimador (``Pipeline`` con su preprocesamiento) para un modelo y una configuración.

    El escalado vive dentro del ``Pipeline``, de modo que en cada ajuste se calcula solo con las filas de
    entrenamiento de ese ajuste y no hay fuga de información hacia la validación ni hacia la prueba.

    Parameters
    ----------
    nombre : str
        Una de las claves de ``ESPACIOS``.
    params : dict
        Hiperparámetros ya decodificados (ver ``optimizadores.decodificar``).
    n_jobs : int
        Núcleos para KNN y Random Forest.
    semilla : int
        Semilla de los modelos con componente aleatoria.
    device : str
        ``"cuda"`` o ``"cpu"`` para XGBoost.

    Returns
    -------
    estimador de scikit-learn sin ajustar.
    """
    p = dict(params)
    if nombre == "KNN":
        return _escalado(KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"],
                                             algorithm="auto", n_jobs=n_jobs))  # con 26 predictoras elige brute
    if nombre in ("Ridge", "Lasso"):
        base = (Ridge(alpha=p["alpha"]) if nombre == "Ridge"
                else Lasso(alpha=p["alpha"], max_iter=5000, tol=1e-3, selection="cyclic"))
        est = _escalado(base)
        return TransformedTargetRegressor(est, func=np.log1p, inverse_func=np.expm1) if p["log_y"] else est
    if nombre == "Árbol de decisión":
        return DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                     max_features=p["max_features"], random_state=semilla)
    if nombre == "Random Forest":
        return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                     n_jobs=n_jobs, random_state=semilla)
    if nombre == "XGBoost":
        return xgb.XGBRegressor(tree_method="hist", device=device, random_state=semilla, n_jobs=n_jobs,
                                **{k: p[k] for k in ESPACIOS["XGBoost"]})
    if nombre == "SVR":
        # SVR lineal resuelto en el primal (O(n·p)); el núcleo RBF se aproxima con Nyström (300 componentes)
        # para evitar el costo cuadrático o cúbico del núcleo completo con cientos de miles de filas
        svr = LinearSVR(C=p["C"], epsilon=p["epsilon"], loss="squared_epsilon_insensitive", dual=False,
                        max_iter=5000, random_state=semilla)
        if p["kernel"] == "lineal":
            return _escalado(svr)
        return Pipeline([("escalado", StandardScaler()),
                         ("nystroem", Nystroem(gamma=p["gamma"], n_components=300, random_state=semilla)),
                         ("modelo", svr)])
    raise ValueError(nombre)

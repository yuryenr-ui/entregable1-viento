"""Rotation Forest para regresión (Rodríguez, Kuncheva y Alonso, 2006; versión de regresión de Pardo et al., 2013).

Cada árbol se entrena con las predictoras "rotadas": se reparten al azar en grupos de ``m`` columnas, en cada
grupo se ajusta un análisis de componentes principales (PCA) con una submuestra del 75 % de las filas, y con
todas las cargas se arma una matriz de rotación R (diagonal por bloques). El árbol ve ``X R`` en lugar de ``X``.

La diferencia con Random Forest es de dónde sale la diversidad: en RF cada árbol ve otra muestra de filas y otro
subconjunto de columnas en cada corte; aquí cada árbol ve combinaciones lineales distintas de las columnas, de
modo que sus cortes son oblicuos en el espacio original (por ejemplo, un corte sobre una mezcla de la velocidad
actual y su rezago de 24 h). Se conservan todos los componentes, así que ninguna rotación pierde información.

Las predictoras se estandarizan antes del PCA para que las variables con unidades grandes no dominen las
componentes. scikit-learn no trae este modelo; la construcción de las rotaciones es propia.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeRegressor

from . import SEED


def _rotacion(Xs, m, frac, rng):
    """Matriz de rotación p×p diagonal por bloques a partir de PCA en grupos aleatorios de ``m`` columnas."""
    n, p = Xs.shape
    R = np.zeros((p, p), np.float32)
    orden = rng.permutation(p)
    for grupo in np.array_split(orden, int(np.ceil(p / m))):
        filas = rng.choice(n, size=max(int(frac * n), len(grupo) + 1), replace=False)
        sub = Xs[np.ix_(filas, grupo)]
        sub = sub - sub.mean(0)
        _, _, Vt = np.linalg.svd(sub, full_matrices=False)          # filas de Vt: direcciones principales
        cargas = np.zeros((len(grupo), len(grupo)), np.float32)
        cargas[:, :Vt.shape[0]] = Vt.T
        R[np.ix_(grupo, grupo)] = cargas
    return R


class RotationForest:
    """Bosque de árboles de regresión entrenados sobre rotaciones PCA aleatorias de las predictoras.

    Parameters
    ----------
    n_estimators : int
        Número de árboles.
    m : int
        Columnas por grupo de la rotación (con 26 predictoras y ``m = 4`` hay 7 grupos).
    max_depth, min_samples_leaf, max_features
        Hiperparámetros de cada ``DecisionTreeRegressor``.
    frac : float
        Fracción de filas con la que se estima cada PCA.
    n_filas_pca : int
        Tope de filas para el PCA (con cientos de miles de filas, 20 mil bastan para estimar las cargas).
    """

    def __init__(self, n_estimators=100, m=4, max_depth=None, min_samples_leaf=5, max_features=1.0, frac=0.75,
                 n_filas_pca=20000, n_jobs=16, semilla=SEED):
        self.n_estimators, self.m, self.max_depth = int(n_estimators), int(m), max_depth
        self.min_samples_leaf, self.max_features, self.frac = int(min_samples_leaf), max_features, frac
        self.n_filas_pca, self.n_jobs, self.semilla = n_filas_pca, n_jobs, semilla

    def _uno(self, Xs, y, k):
        rng = np.random.default_rng([self.semilla, k])
        idx = rng.choice(len(Xs), size=min(self.n_filas_pca, len(Xs)), replace=False)
        R = _rotacion(Xs[idx], self.m, self.frac, rng)
        arbol = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                      max_features=self.max_features, random_state=int(rng.integers(2**31)))
        return R, arbol.fit(Xs @ R, y)

    def fit(self, X, y):
        X = np.asarray(X, np.float32)
        self.media, self.escala = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.media) / self.escala
        self.arboles = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._uno)(Xs, np.asarray(y, float), k) for k in range(self.n_estimators))
        return self

    def predict(self, X):
        Xs = (np.asarray(X, np.float32) - self.media) / self.escala
        preds = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(a.predict)(Xs @ R) for R, a in self.arboles)
        return np.mean(preds, 0)

"""Modelo original: regresión por diferencias de pares por estación (PADRE por estación) y su ensamble.

Clases compatibles con scikit-learn (``BaseEstimator`` + ``RegressorMixin``): se pueden clonar, poner en un
``Pipeline`` y usar con ``cross_val_score``. Solo usan numpy y pandas; el regresor interno ``g`` que aprende las
diferencias es un parámetro (cualquier regresor de scikit-learn, o XGBoost).

Algoritmo (PADRE por estación)
------------------------------
Entrada: filas ``(x_i, s_i, y_i)`` de entrenamiento, con ``s_i`` la estación; ``P`` pares por fila; ``M`` anclas;
``n_min`` filas mínimas por estación; regresor ``g``.

Entrenamiento:
    1. Para cada estación ``s``, su fuente ``F(s)`` es ``s`` si tiene al menos ``n_min`` filas y, si no, la
       estación más cercana (distancia de haversine) que sí las tiene.
    2. Para cada fila ``i`` y ``k = 1, …, P``, sortear ``j`` al azar entre las filas de ``F(s_i)``.
    3. Ajustar ``g`` sobre las ``P·n`` filas ``z_ij = (x_i, x_i − x_j)`` con objetivo ``y_i − y_j``
       (pérdida cuadrática).
Predicción de ``(x, s)``:
    4. Sortear ``M`` anclas ``j_1, …, j_M`` entre las filas de ``F(s)``.
    5. ``ŷ = (1/M) Σ_m [ y_{j_m} + g(x, x − x_{j_m}) ]``.
Criterio de parada: el algoritmo no itera; termina tras un único ajuste de ``g`` (cuya parada es la de ``g``).

Complejidad: si ajustar ``g`` con ``n`` filas y ``2p`` columnas cuesta ``T_g(n, 2p)``, el entrenamiento cuesta
``O(P·n·p) + T_g(P·n, 2p)`` y la predicción de ``q`` filas ``O(M·q·p)`` más ``M·q`` evaluaciones de ``g``.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import SEED


def _haversine(lat, lon):
    """Matriz de distancias (km) entre todos los pares de puntos."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    h = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


class PadreEstacionRegresor(BaseEstimator, RegressorMixin):
    """PADRE con pares y anclas de la misma estación.

    La columna ``col_estacion`` de ``X`` es el código numérico de la estación; las demás son las predictoras.

    Parameters
    ----------
    estimador : regresor de scikit-learn, opcional
        Regresor ``g`` que aprende las diferencias; se clona en cada ``fit``. Por defecto, regresión lineal.
    pares : int
        Parejas sorteadas por fila de entrenamiento (``P``).
    anclas : int
        Anclas promediadas por pronóstico (``M``).
    min_filas : int
        Filas mínimas para que una estación use sus propios pares y anclas (``n_min``).
    coords : pandas.DataFrame, opcional
        Índice: código de estación (numérico o texto numérico); columnas ``lat`` y ``lon``. Sin coordenadas, una
        estación con pocas filas usa la estación con más filas.
    col_estacion : int
        Posición de la columna con el código de la estación.
    random_state : int
        Semilla de los sorteos de pares (``random_state``) y de anclas (``random_state + 1``).
    """

    def __init__(self, estimador=None, pares=4, anclas=20, min_filas=500, coords=None, col_estacion=0,
                 random_state=SEED):
        self.estimador = estimador
        self.pares = pares
        self.anclas = anclas
        self.min_filas = min_filas
        self.coords = coords
        self.col_estacion = col_estacion
        self.random_state = random_state

    # --- utilidades -------------------------------------------------------------------------------------------
    def _separar(self, X):
        """Devuelve (predictoras en float32, código de estación)."""
        est = X[:, self.col_estacion]
        return np.delete(X, self.col_estacion, axis=1).astype(np.float32), est

    def _fuente(self, e):
        """Estación de la que salen los pares y anclas de ``e``: ella misma o, si tiene pocas filas, una vecina."""
        if self.conteo_.get(e, 0) >= self.min_filas:
            return e
        if self.dist_ is None or e not in self.dist_.index:
            if self.conteo_.get(e, 0) and self.dist_ is None and max(self.conteo_.values()) < self.min_filas:
                return e
            return max(self.conteo_, key=self.conteo_.get)
        orden = [s for s in self.dist_.loc[e].sort_values().index if s != e]
        # la vecina más cercana con datos suficientes; si ninguna llega al mínimo, la propia estación si tiene
        # filas y, si no, la vecina más cercana con alguna fila
        return next((s for s in orden if self.conteo_.get(s, 0) >= self.min_filas),
                    e if self.conteo_.get(e, 0) else next(s for s in orden if self.conteo_.get(s, 0)))

    def _parejas(self, est, rng):
        """Para cada fila sortea una posición del entrenamiento dentro de la estación fuente de su estación."""
        j = np.empty(len(est), dtype=np.int64)
        for e in np.unique(est):
            m = est == e
            j[m] = rng.choice(self.filas_[self._fuente(e)], m.sum())
        return j

    # --- API de scikit-learn ----------------------------------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        if X.shape[1] < 2:
            raise ValueError("X necesita la columna de estación y al menos una predictora.")
        if int(self.pares) < 1 or int(self.anclas) < 1:
            raise ValueError("pares y anclas deben ser enteros positivos.")
        rng = np.random.default_rng(self.random_state)
        self.X_, est = self._separar(X)
        self.y_ = y.astype(np.float32)
        self.n_features_in_ = X.shape[1]
        self.conteo_ = pd.Series(est).value_counts().to_dict()
        self.filas_ = {e: np.flatnonzero(est == e) for e in self.conteo_}
        self.dist_ = None
        if self.coords is not None:
            co = self.coords.copy()
            co.index = co.index.astype(float)
            self.dist_ = pd.DataFrame(_haversine(co.lat.to_numpy(), co.lon.to_numpy()), index=co.index,
                                      columns=co.index)
        i = np.repeat(np.arange(len(self.y_)), int(self.pares))
        j = self._parejas(est[i], rng)
        self.estimador_ = clone(self.estimador) if self.estimador is not None else LinearRegression()
        self.estimador_.fit(np.hstack([self.X_[i], self.X_[i] - self.X_[j]]), self.y_[i] - self.y_[j])
        return self

    def predict(self, X):
        check_is_fitted(self, "estimador_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; el modelo se ajustó con {self.n_features_in_}.")
        rng = np.random.default_rng(self.random_state + 1)
        Xp, est = self._separar(X)
        M = int(self.anclas)
        iv = np.repeat(np.arange(len(Xp)), M)
        jv = self._parejas(est[iv], rng)
        d = self.estimador_.predict(np.hstack([Xp[iv], Xp[iv] - self.X_[jv]]))
        return (self.y_[jv] + d).reshape(len(Xp), M).mean(1)


class EnsamblePadre(BaseEstimator, RegressorMixin):
    """Promedio simple de regresores del curso y de modelos PADRE por estación.

    Los componentes ``PadreEstacionRegresor`` reciben ``X`` completa; los demás, ``X`` sin la columna de estación.

    Parameters
    ----------
    estimadores : list of (str, estimador)
        Componentes del promedio; se clonan en cada ``fit``.
    col_estacion : int
        Posición de la columna con el código de la estación.
    """

    def __init__(self, estimadores, col_estacion=0):
        self.estimadores = estimadores
        self.col_estacion = col_estacion

    def _entrada(self, est, X):
        # los modelos del curso reciben las predictoras en float32, igual que en su búsqueda
        if isinstance(est, PadreEstacionRegresor):
            return X
        return np.delete(X, self.col_estacion, axis=1).astype(np.float32)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        if not self.estimadores:
            raise ValueError("El ensamble necesita al menos un componente.")
        self.n_features_in_ = X.shape[1]
        self.estimadores_ = [(n, clone(e).fit(self._entrada(e, X), y)) for n, e in self.estimadores]
        return self

    def predict_componentes(self, X):
        """Pronóstico de cada componente (columnas en el orden de ``estimadores``)."""
        check_is_fitted(self, "estimadores_")
        X = check_array(X, dtype=np.float64)
        return pd.DataFrame({n: e.predict(self._entrada(e, X)) for n, e in self.estimadores_})

    def predict(self, X):
        return self.predict_componentes(X).to_numpy().mean(1)

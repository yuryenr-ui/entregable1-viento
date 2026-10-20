"""Modelo propuesto: PADRE por estación combinado con XGBoost.

PADRE (Pairwise Difference Regression; Tynes et al., J. Chem. Inf. Model. 2021) entrena un regresor sobre
PARES de filas para predecir la diferencia ``y_i − y_j`` a partir de ``(x_i, x_i − x_j)``. Para pronosticar una
fila nueva ``x`` se eligen ``M`` anclas ``j`` del entrenamiento, cuyo valor observado se conoce, y se promedia
``y_j + f(x, x − x_j)``.

La variante por estación, propia de este trabajo, forma los pares y elige las anclas dentro de la misma
estación: el regresor aprende cuánto se aparta una hora de otras horas conocidas de ese sitio, y lo que es fijo
de la estación (altitud, rugosidad, exposición) se cancela en la diferencia. Una estación con menos de
``min_filas`` filas de entrenamiento toma sus pares y anclas de la estación vecina más cercana con datos.

El pronóstico final mezcla PADRE con el XGBoost del Entregable 3: ``w · PADRE + (1 − w) · XGBoost``, con ``w``
elegido en la validación interna.
"""
import numpy as np
import pandas as pd

from . import SEED
from .modelos import construir
from .multitarea import haversine


class PadreEstacion:
    """PADRE con pares y anclas de la misma estación.

    Parameters
    ----------
    params_motor : dict
        Hiperparámetros del regresor que aprende las diferencias (claves de ``modelos.ESPACIOS[motor]``).
    coords : pandas.DataFrame
        Índice: código de estación; columnas ``lat`` y ``lon`` (para el respaldo de la vecina).
    pares : int
        Parejas por fila de entrenamiento (el conjunto de pares tiene ``pares`` veces más filas).
    anclas : int
        Anclas promediadas por pronóstico.
    motor : str
        ``"XGBoost"`` o ``"Random Forest"``.
    min_filas : int
        Filas mínimas para que una estación use sus propias anclas.
    """

    def __init__(self, params_motor, coords, pares=4, anclas=20, motor="XGBoost", min_filas=500,
                 n_jobs=8, semilla=SEED):
        self.params_motor, self.pares, self.anclas = params_motor, int(pares), int(anclas)
        self.motor, self.min_filas, self.n_jobs, self.semilla = motor, min_filas, n_jobs, semilla
        self.dist = pd.DataFrame(haversine(coords.lat.to_numpy(), coords.lon.to_numpy()),
                                 index=coords.index, columns=coords.index)

    def _fuente(self, e):
        """Estación de la que salen las parejas de ``e``: ella misma o, si tiene pocas filas, su vecina."""
        if self.conteo.get(e, 0) >= self.min_filas:
            return e
        orden = [s for s in self.dist.loc[e].sort_values().index if s != e]
        # la vecina más cercana con datos suficientes; si ninguna llega al mínimo (submuestras pequeñas), la
        # propia estación si tiene filas y, si no, la vecina más cercana con alguna fila
        return next((s for s in orden if self.conteo.get(s, 0) >= self.min_filas),
                    e if self.conteo.get(e, 0) else next(s for s in orden if self.conteo.get(s, 0)))

    def _parejas(self, est, rng):
        """Para cada estación de ``est`` sortea posiciones del entrenamiento dentro de su estación fuente."""
        j = np.empty(len(est), dtype=np.int64)
        for e in np.unique(est):
            m = est == e
            j[m] = rng.choice(self.filas[self._fuente(e)], m.sum())
        return j

    def fit(self, X, y, est):
        rng = np.random.default_rng(self.semilla)
        self.X, self.y = np.asarray(X, np.float32), np.asarray(y, np.float32)
        est = np.asarray(est)
        self.conteo = pd.Series(est).value_counts().to_dict()
        self.filas = {e: np.flatnonzero(est == e) for e in self.conteo}
        i = np.repeat(np.arange(len(self.y)), self.pares)
        j = self._parejas(est[i], rng)
        self.modelo = construir(self.motor, self.params_motor, n_jobs=self.n_jobs, semilla=self.semilla)
        self.modelo.fit(np.hstack([self.X[i], self.X[i] - self.X[j]]), self.y[i] - self.y[j])
        return self

    def predict(self, X, est):
        rng = np.random.default_rng(self.semilla + 1)
        X = np.asarray(X, np.float32)
        iv = np.repeat(np.arange(len(X)), self.anclas)
        jv = self._parejas(np.asarray(est)[iv], rng)
        d = self.modelo.predict(np.hstack([X[iv], X[iv] - self.X[jv]]))
        return (self.y[jv] + d).reshape(len(X), self.anclas).mean(1)


# Espacio de búsqueda: el del motor XGBoost más los dos hiperparámetros propios de PADRE
ESPACIO_PADRE = {"pares": ("cat", [2, 4, 8]), "anclas": ("cat", [10, 20, 40])}
PESOS = np.round(np.linspace(0, 1, 21), 2)   # w de la mezcla w · PADRE + (1 − w) · XGBoost


def mejor_peso(y, p_padre, p_xgb):
    """Peso ``w`` de la rejilla ``PESOS`` con menor RMSE y ese RMSE. ``y`` y los pronósticos son listas por pliegue."""
    err = [np.mean([np.sqrt(np.mean((yy - (w * pp + (1 - w) * px)) ** 2))
                    for yy, pp, px in zip(y, p_padre, p_xgb)]) for w in PESOS]
    k = int(np.argmin(err))
    return float(PESOS[k]), float(err[k])

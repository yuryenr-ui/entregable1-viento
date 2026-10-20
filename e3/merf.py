"""Random Forest de efectos mixtos (MERF; Hajjem, Bellavance y Larocque, 2014) con una tarea por estación.

El modelo separa una parte común a todas las estaciones, aprendida por un Random Forest, de un ajuste lineal
propio de cada estación (efecto aleatorio):

    y_si = f(x_si) + z_siᵀ b_s + ε_si,      b_s ~ N(0, D),      ε_si ~ N(0, σ²)

donde ``z`` son unas pocas columnas (intercepto, velocidad actual, ciclo diario) cuyo efecto puede variar entre
estaciones. Se estima con el algoritmo EM:

1. con los efectos actuales, el bosque se ajusta a ``y − zᵀb`` (la parte que no explica la estación);
2. con el residuo fuera de bolsa del bosque, cada ``b_s`` es la media a posteriori
   ``b_s = C_s Z_sᵀ r_s / σ²`` con ``C_s = (Z_sᵀZ_s / σ² + D⁻¹)⁻¹``, que encoge hacia cero los efectos de las
   estaciones con pocos datos;
3. se actualizan ``σ²`` y ``D`` con las expresiones del paso M.

Usamos el pronóstico fuera de bolsa (OOB) del bosque en el paso 2: con el pronóstico dentro de la muestra, el
bosque ya memoriza parte de cada estación y los efectos aleatorios quedarían subestimados. Una estación sin datos
de entrenamiento recibe ``b = 0``, es decir, solo la parte común.

Opción ``ventana_dias``: el efecto de cada estación se estima solo con sus últimos ``ventana_dias`` días de
entrenamiento (el bosque sigue usando todas las filas). Permite que el ajuste por estación siga los cambios de
sensor o de entorno en lugar de promediar toda la historia; la validación interna decide si se usa.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import SEED

# columnas con efecto aleatorio por estación, de menos a más flexible
EFECTOS = {"intercepto": [], "intercepto + vel_l0": ["vel_l0"],
           "intercepto + vel_l0 + ciclo diario": ["vel_l0", "hy_sin", "hy_cos"]}


class MERF:
    """Random Forest de efectos mixtos por estación.

    Parameters
    ----------
    params_rf : dict
        Hiperparámetros del ``RandomForestRegressor`` (claves de ``modelos.ESPACIOS["Random Forest"]``).
    columnas : list of str
        Nombres de las predictoras, en el orden de las columnas de ``X``.
    efectos : str
        Clave de ``EFECTOS``: columnas cuyo coeficiente varía entre estaciones (además del intercepto).
    iteraciones : int
        Iteraciones del algoritmo EM.
    ventana_dias : int or None
        Días más recientes de cada estación con los que se estima su efecto (``None``: toda la historia).
    """

    def __init__(self, params_rf, columnas, efectos="intercepto + vel_l0 + ciclo diario", iteraciones=4,
                 ventana_dias=None, n_jobs=8, semilla=SEED):
        self.params_rf, self.columnas = dict(params_rf), list(columnas)
        self.efectos, self.iteraciones, self.n_jobs, self.semilla = efectos, int(iteraciones), n_jobs, semilla
        self.ventana_dias = ventana_dias
        self.cols_z = [self.columnas.index(c) for c in EFECTOS[efectos]]

    def _z(self, X):
        Z = np.column_stack([np.ones(len(X)), X[:, self.cols_z]]) if self.cols_z else np.ones((len(X), 1))
        return (Z - self.centro) / self.escala

    def _bosque(self):
        p = self.params_rf
        return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                     oob_score=True, n_jobs=self.n_jobs, random_state=self.semilla)

    def fit(self, X, y, est, t=None):
        """Ajusta el modelo; ``t`` (fecha de cada fila) solo se necesita si ``ventana_dias`` no es ``None``."""
        X, y, est = np.asarray(X, np.float32), np.asarray(y, float), np.asarray(est)
        Z0 = np.column_stack([np.ones(len(X)), X[:, self.cols_z]]) if self.cols_z else np.ones((len(X), 1))
        self.centro = np.r_[0.0, Z0[:, 1:].mean(0)] if self.cols_z else np.zeros(1)
        self.escala = np.r_[1.0, Z0[:, 1:].std(0) + 1e-9] if self.cols_z else np.ones(1)
        Z = self._z(X)
        q = Z.shape[1]
        grupos = {s: np.flatnonzero(est == s) for s in np.unique(est)}
        if self.ventana_dias is not None:          # filas recientes de cada estación para estimar su efecto
            t = np.asarray(t, "datetime64[ns]")
            limite = np.timedelta64(int(self.ventana_dias), "D")
            recientes = {s: i[t[i] >= t[i].max() - limite] for s, i in grupos.items()}
        else:
            recientes = grupos
        b = {s: np.zeros(q) for s in grupos}
        D, s2 = np.eye(q) * 0.1, float(np.var(y))
        self.historia = []
        for _ in range(self.iteraciones):
            efecto = np.zeros(len(y))
            for s, i in grupos.items():
                efecto[i] = Z[i] @ b[s]
            self.bosque = self._bosque().fit(X, y - efecto)
            f = self.bosque.oob_prediction_
            f = np.where(np.isfinite(f), f, self.bosque.predict(X))     # filas sin predicción fuera de bolsa
            r = y - f
            D_inv = np.linalg.inv(D)
            suma_e, suma_tr, suma_D, n_usadas = 0.0, 0.0, np.zeros((q, q)), 0
            for s, i in recientes.items():
                n_usadas += len(i)
                Zs, rs = Z[i], r[i]
                C = np.linalg.inv(Zs.T @ Zs / s2 + D_inv)                # covarianza a posteriori de b_s
                b[s] = C @ Zs.T @ rs / s2
                e = rs - Zs @ b[s]
                suma_e += e @ e
                suma_tr += np.trace(Zs @ C @ Zs.T)                         # = n_s σ² − σ⁴ tr(V_s⁻¹)
                suma_D += np.outer(b[s], b[s]) + C
            s2 = (suma_e + suma_tr) / n_usadas
            D = suma_D / len(grupos)
            self.historia.append({"sigma2": s2, "D_diag": np.diag(D).tolist()})
        self.b, self.D, self.sigma2 = b, D, s2
        return self

    def predict(self, X, est):
        X, est = np.asarray(X, np.float32), np.asarray(est)
        Z = self._z(X)
        p = self.bosque.predict(X)
        for s in np.unique(est):
            if s in self.b:
                i = est == s
                p[i] += Z[i] @ self.b[s]
        return p

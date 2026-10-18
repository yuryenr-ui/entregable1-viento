"""Modelo propuesto: modelo global + corrección multi-tarea por estación regularizada con un grafo geográfico.

Cada estación es una tarea. Sobre el pronóstico de un modelo global ``g(x)`` (XGBoost o Random Forest), cada
estación ``s`` aprende una corrección lineal ``v_s · z`` con ``z = (1, g(x), x)`` estandarizado. Las correcciones se
estiman juntas minimizando

    Σ_s ||r_s − Z_s v_s||² + λ₁ Σ_s ||v_s||² + λ₂ Σ_{s<t} a_st ||v_s − v_t||²

donde ``r = y − g(x)`` es el residuo del modelo global y ``a_st`` el peso de la arista entre dos estaciones: un
núcleo gaussiano de la distancia geográfica, restringido a los ``k`` vecinos más cercanos. Con ``λ₂ = 0`` cada
estación se corrige por separado; con ``λ₂ → ∞`` todas comparten la misma corrección. El grafo intermedio deja que
las estaciones con pocos datos tomen prestada información de sus vecinas, y una estación sin datos de entrenamiento
(por ejemplo, una que entra a la red en 2025) recibe el promedio ponderado de las correcciones de sus vecinas.

La solución es cerrada: el sistema lineal ``(blockdiag(Z_sᵀZ_s) + λ₁ I + λ₂ L ⊗ I) v = vec(Z_sᵀ r_s)``, con ``L`` el
laplaciano del grafo, de tamaño ``S·q`` (unas 60 estaciones × 28 coeficientes).
"""
import numpy as np
from scipy.linalg import solve


def haversine(lat, lon):
    """Matriz de distancias (km) entre todos los pares de puntos."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    h = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def grafo(lat, lon, k=4, escala_km=50.0):
    """Pesos simétricos ``a_st = exp(−d²/2σ²)`` entre cada estación y sus ``k`` vecinas más cercanas."""
    d = haversine(np.asarray(lat), np.asarray(lon))
    np.fill_diagonal(d, np.inf)
    A = np.zeros_like(d)
    for i in range(len(d)):
        j = np.argsort(d[i])[:k]
        A[i, j] = np.exp(-d[i, j] ** 2 / (2 * escala_km ** 2))
    return np.maximum(A, A.T)


class CorreccionMultitarea:
    """Corrección lineal por estación con regularización de grafo (ver el docstring del módulo).

    Parameters
    ----------
    estaciones : sequence of str
        Todas las estaciones del grafo (también las que no tienen datos de entrenamiento).
    lat, lon : array-like
        Coordenadas de cada estación, en el mismo orden.
    lam1, lam2 : float
        Penalización de la magnitud de cada corrección y de la diferencia entre estaciones vecinas.
    k, escala_km : int, float
        Vecinos por estación y escala del núcleo gaussiano de distancia.
    """

    def __init__(self, estaciones, lat, lon, lam1=100.0, lam2=1000.0, k=4, escala_km=50.0):
        self.estaciones = list(estaciones)
        self.pos = {e: i for i, e in enumerate(self.estaciones)}
        self.A = grafo(lat, lon, k, escala_km)
        self.lam1, self.lam2 = lam1, lam2

    def _z(self, X, g):
        Z = np.column_stack([g, X])
        return np.column_stack([np.ones(len(Z)), (Z - self.media) / self.desv])

    def fit(self, X, g, y, est):
        """Ajusta las correcciones con predictoras ``X``, pronóstico global ``g``, observado ``y`` y estación ``est``."""
        Z0 = np.column_stack([g, X])
        self.media, self.desv = Z0.mean(0), Z0.std(0) + 1e-9
        Z = self._z(X, g)
        r = np.asarray(y) - np.asarray(g)
        S, q = len(self.estaciones), Z.shape[1]
        M = np.zeros((S * q, S * q))
        b = np.zeros(S * q)
        idx = np.array([self.pos[e] for e in est])
        for s in np.unique(idx):
            m = idx == s
            Zs = Z[m]
            M[s * q:(s + 1) * q, s * q:(s + 1) * q] = Zs.T @ Zs
            b[s * q:(s + 1) * q] = Zs.T @ r[m]
        L = np.diag(self.A.sum(1)) - self.A
        M += self.lam1 * np.eye(S * q) + self.lam2 * np.kron(L, np.eye(q))
        self.V = solve(M, b, assume_a="pos").reshape(S, q)   # simétrica definida positiva
        return self

    def predict(self, X, g, est):
        """Pronóstico corregido ``g + v_s · z``."""
        Z = self._z(X, g)
        idx = np.array([self.pos[e] for e in est])
        return np.asarray(g) + np.einsum("ij,ij->i", Z, self.V[idx])

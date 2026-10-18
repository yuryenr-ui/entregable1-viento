"""Carga del conjunto de modelado y pliegues cronológicos para la validación anidada."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import SEED

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados" / "e3"
TEST_INI = pd.Timestamp("2025-01-01")
TRIMESTRES = pd.date_range("2023-01-01", "2025-01-01", freq="QS")  # inicios de trimestre 2023-T1 ... 2025-T1


def cargar():
    """Lee el conjunto de modelado guardado por ``scripts/e3_preparar_datos.py``.

    Returns
    -------
    TR, TE : pandas.DataFrame
        Entrenamiento (objetivo anterior a 2025) y prueba (pronósticos emitidos en 2025).
    F : list of str
        Las 26 predictoras de la sección 2.0.
    """
    D = pd.read_parquet(RESULTADOS / "modelado.parquet")
    F = (RESULTADOS / "predictoras.txt").read_text(encoding="utf-8").split("\n")
    return D[D.t_y < TEST_INI].reset_index(drop=True), D[D.t >= TEST_INI].reset_index(drop=True), F


def corte(datos, ini, fin):
    """Índices de entrenamiento y validación para validar en el periodo ``[ini, fin)``.

    El entrenamiento solo usa filas cuyo objetivo es anterior a ``ini`` (ventana creciente con separación
    de 24 h) y la validación, pronósticos emitidos en el periodo cuyo objetivo también cae dentro de él.

    Parameters
    ----------
    datos : pandas.DataFrame
        Debe tener las columnas ``t`` (emisión) y ``t_y`` (fecha del objetivo).
    ini, fin : pandas.Timestamp
        Inicio y fin del periodo de validación.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Posiciones de entrenamiento y de validación dentro de ``datos``.
    """
    tr = np.flatnonzero((datos.t_y < ini).to_numpy())
    va = np.flatnonzero(((datos.t >= ini) & (datos.t_y < fin)).to_numpy())
    return tr, va


def pliegues_externos(TR):
    """Bucle externo: los 4 trimestres de 2024, cada uno con todo lo anterior como entrenamiento.

    Returns
    -------
    list of (str, numpy.ndarray, numpy.ndarray)
        Nombre del pliegue, posiciones de entrenamiento y de validación en ``TR``.
    """
    out = []
    for k in range(4):
        ini, fin = TRIMESTRES[4 + k], TRIMESTRES[5 + k]
        out.append((f"2024-T{k + 1}", *corte(TR, ini, fin)))
    return out


def pliegues_internos(TR, fin_entrenamiento, n=3):
    """Bucle interno: los ``n`` trimestres inmediatamente anteriores a ``fin_entrenamiento``.

    Se aplica sobre el entrenamiento de un pliegue externo, de modo que la búsqueda de hiperparámetros nunca
    ve el trimestre con el que luego se mide el desempeño. Para la evaluación final en 2025 se usa
    ``fin_entrenamiento = 2025-01-01`` y ``n = 4``, que son los trimestres de 2024 de la sección 3.1.
    """
    pos = TRIMESTRES.get_loc(fin_entrenamiento)
    return [corte(TR, TRIMESTRES[i], TRIMESTRES[i + 1]) for i in range(pos - n, pos)]


def submuestra(indices, n, semilla):
    """Submuestra reproducible de ``n`` posiciones (todas si hay menos), en orden cronológico."""
    if len(indices) <= n:
        return indices
    return np.sort(np.random.default_rng(semilla).choice(indices, n, replace=False))


def particion_interna(TR, fin_entrenamiento, n_pliegues, n_ajuste, n_valid, semilla=SEED):
    """Pliegues internos ya submuestreados, iguales para todas las configuraciones de una búsqueda.

    Usar las mismas filas para todas las configuraciones hace que las comparaciones entre ellas sean
    pareadas, y la submuestra mantiene el costo de cada evaluación acotado (fidelidad baja); la configuración
    elegida se reentrena después con todo el entrenamiento.
    """
    return [(submuestra(tr, n_ajuste, semilla + i), submuestra(va, n_valid, semilla + 100 + i))
            for i, (tr, va) in enumerate(pliegues_internos(TR, fin_entrenamiento, n_pliegues))]

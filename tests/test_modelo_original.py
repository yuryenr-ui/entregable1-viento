"""Pruebas unitarias del modelo original (``e3/modelo_original.py``).

Ejecutar desde la carpeta entregable1:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3.modelo_original import EnsamblePadre, PadreEstacionRegresor  # noqa: E402

ESTACIONES = np.array([101.0, 102.0, 103.0])
NIVEL = {101.0: 0.0, 102.0: 3.0, 103.0: -2.0}          # efecto fijo α_s de cada estación
COORDS = pd.DataFrame({"lat": [10.0, 10.1, 11.0], "lon": [-75.0, -75.1, -74.0]}, index=ESTACIONES)


def datos_sinteticos(n=900, ruido=0.1, semilla=0):
    """y = 2·x1 − x2 + α_s + ε: la solución es conocida."""
    rng = np.random.default_rng(semilla)
    est = rng.choice(ESTACIONES, n)
    x = rng.normal(size=(n, 2))
    y = 2 * x[:, 0] - x[:, 1] + np.vectorize(NIVEL.get)(est) + rng.normal(scale=ruido, size=n)
    return np.column_stack([est, x]), y


def test_aprende_solucion_conocida():
    """(i) Con g lineal recupera la relación y el nivel de cada estación: el error queda cerca del ruido."""
    X, y = datos_sinteticos()
    Xp, yp = datos_sinteticos(n=300, semilla=1)
    m = PadreEstacionRegresor(LinearRegression(), pares=4, anclas=20, min_filas=10).fit(X, y)
    rmse = np.sqrt(np.mean((m.predict(Xp) - yp) ** 2))
    assert rmse < 0.2
    # el efecto de la estación se cancela en las diferencias: g usa las pendientes 2 y −1 sobre x_i − x_j
    coef = m.estimador_.coef_
    assert np.allclose(coef[2:], [2, -1], atol=0.05)


def test_supera_modelo_global_sin_estacion():
    """El mismo regresor lineal sin la estación no puede representar α_s; PADRE por estación sí."""
    X, y = datos_sinteticos()
    Xp, yp = datos_sinteticos(n=300, semilla=1)
    global_ = LinearRegression().fit(X[:, 1:], y)
    padre = PadreEstacionRegresor(LinearRegression(), min_filas=10).fit(X, y)
    err = lambda p: np.sqrt(np.mean((p - yp) ** 2))
    assert err(padre.predict(Xp)) < 0.5 * err(global_.predict(Xp[:, 1:]))


def test_reproducible_con_la_misma_semilla():
    """(ii) Misma semilla, mismas predicciones; otra semilla, otros sorteos."""
    X, y = datos_sinteticos()
    a = PadreEstacionRegresor(DecisionTreeRegressor(random_state=0), random_state=7).fit(X, y).predict(X[:50])
    b = PadreEstacionRegresor(DecisionTreeRegressor(random_state=0), random_state=7).fit(X, y).predict(X[:50])
    c = PadreEstacionRegresor(DecisionTreeRegressor(random_state=0), random_state=8).fit(X, y).predict(X[:50])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("malo", [np.nan, np.inf])
def test_rechaza_valores_invalidos(malo):
    """(iii) NaN o infinitos en X se rechazan con un error claro."""
    X, y = datos_sinteticos(n=60)
    X[3, 1] = malo
    with pytest.raises(ValueError):
        PadreEstacionRegresor(min_filas=5).fit(X, y)


def test_rechaza_entradas_mal_formadas():
    """(iii) Predecir sin ajustar, cambiar el número de columnas o hiperparámetros no válidos."""
    X, y = datos_sinteticos(n=60)
    with pytest.raises(NotFittedError):
        PadreEstacionRegresor().predict(X)
    m = PadreEstacionRegresor(min_filas=5).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :2])
    with pytest.raises(ValueError):
        PadreEstacionRegresor(pares=0).fit(X, y)
    with pytest.raises(ValueError):
        PadreEstacionRegresor().fit(X[:, :1], y)
    with pytest.raises(ValueError):
        PadreEstacionRegresor().fit(X, y[:-1])


def test_estacion_nueva_y_estacion_pequena_usan_la_vecina():
    """Una estación sin filas suficientes (o desconocida) toma pares y anclas de la vecina más cercana."""
    X, y = datos_sinteticos()
    m = PadreEstacionRegresor(min_filas=10, coords=COORDS).fit(X[X[:, 0] != 102.0], y[X[:, 0] != 102.0])
    assert m._fuente(102.0) == 101.0          # 102 está a ~15 km de 101 y a ~150 km de 103
    Xn = X[X[:, 0] == 102.0][:20]
    assert np.isfinite(m.predict(Xn)).all()


def test_compatible_con_clone_pipeline_y_cross_val_score():
    """(iv) clone conserva los hiperparámetros sin copiar el ajuste; funciona en Pipeline y cross_val_score."""
    m = PadreEstacionRegresor(DecisionTreeRegressor(max_depth=3), pares=2, anclas=5, min_filas=10)
    c = clone(m)
    assert c.get_params()["pares"] == 2 and not hasattr(c, "estimador_")
    X, y = datos_sinteticos(n=300)
    s = cross_val_score(Pipeline([("padre", m)]), X, y, cv=3, scoring="neg_root_mean_squared_error")
    assert np.isfinite(s).all()


def test_ensamble_es_el_promedio_de_sus_componentes():
    """El ensamble promedia sin pesos; los modelos globales no reciben la columna de estación."""
    X, y = datos_sinteticos()
    ens = EnsamblePadre([("lineal", LinearRegression()),
                         ("padre", PadreEstacionRegresor(LinearRegression(), min_filas=10))]).fit(X, y)
    comp = ens.predict_componentes(X[:30])
    assert np.allclose(ens.predict(X[:30]), comp.mean(axis=1))
    assert ens.estimadores_[0][1].n_features_in_ == 2
    assert clone(ens).get_params()["estimadores"][1][0] == "padre"

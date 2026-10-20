"""Primera prueba de PADRE (Pairwise Difference Regression, Tynes et al. 2021) con XGBoost en 2024.

PADRE entrena el modelo sobre PARES de filas para predecir la diferencia ``y_i − y_j`` a partir de
``(x_i, x_i − x_j)``. Para pronosticar una fila nueva ``x`` se eligen ``M`` anclas ``j`` del entrenamiento y se
promedia ``y_j + f(x, x − x_j)``. Variantes probadas:

- ``aleatorio``: pares y anclas al azar entre todas las estaciones (PADRE original);
- ``estacion``: pares y anclas de la misma estación (o de la vecina más cercana si tiene pocos datos), de modo
  que el modelo aprende cuánto se aparta la hora a pronosticar de otras horas conocidas de esa estación;
- ``vecinas``: pares y anclas de la estación y de sus 3 vecinas geográficas más cercanas;
- ``reciente``: como ``estacion``, pero la pareja de cada fila sale de las ~90 días anteriores de su estación, y
  las anclas de un pronóstico, de los últimos ~90 días del entrenamiento.

La primera corrida (``padre_exploracion_v1.csv``) incluía la variante ``aleatorio`` y no tenía el respaldo de la
vecina: una estación con 11 filas de entrenamiento disparó el RMSE de 2024-T3.

El motor (el regresor que aprende las diferencias) es XGBoost por defecto o Random Forest si se pasa como
argumento; se compara con el mismo modelo directo del pliegue. 2025 no se usa.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                          # noqa: E402
from e3.datos import RESULTADOS, cargar, pliegues_externos  # noqa: E402
from e3.modelos import construir                            # noqa: E402
from e3.multitarea import haversine                        # noqa: E402

warnings.filterwarnings("ignore")
PARES_POR_FILA, ANCLAS = 4, 20
TR, TE, F = cargar()
X, y, est, t = TR[F].to_numpy(np.float32), TR.y.to_numpy(np.float32), TR.est.to_numpy(), TR.t.to_numpy()
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
D_KM = pd.DataFrame(haversine(co.lat.to_numpy(), co.lon.to_numpy()), index=co.index, columns=co.index)
MIN_FILAS, VECINAS, VENTANA = 500, 3, 2160   # VENTANA: ~90 días de filas horarias


def grupo(e, conteo, variante):
    """Estaciones de las que salen las parejas de ``e``: ella misma y, según la variante, sus vecinas.

    Una estación con menos de ``MIN_FILAS`` filas de entrenamiento usa solo sus vecinas más cercanas con datos.
    """
    con_datos = [s for s in D_KM.loc[e].sort_values().index if s != e and conteo.get(s, 0) >= MIN_FILAS]
    propia = [e] if conteo.get(e, 0) >= MIN_FILAS else []
    if variante == "estacion":
        return propia or con_datos[:1]
    return propia + con_datos[:VECINAS]          # "vecinas"


def pareja(idx_i, pool, variante, rng):
    """Para cada fila de ``idx_i`` elige una pareja dentro de ``pool`` (al azar o en su grupo de estaciones)."""
    if variante == "aleatorio":
        return rng.choice(pool, len(idx_i))
    j = np.empty(len(idx_i), dtype=np.int64)
    est_pool = est[pool]
    conteo = pd.Series(est_pool).value_counts().to_dict()
    for e in np.unique(est[idx_i]):
        m = est[idx_i] == e
        cand = pool[np.isin(est_pool, grupo(e, conteo, "estacion" if variante == "reciente" else variante))]
        if variante != "reciente":
            j[m] = rng.choice(cand, m.sum())
            continue
        # pareja entre las RECIENTES filas anteriores de su grupo (hasta VENTANA filas antes de la fila i)
        cand = cand[np.argsort(t[cand], kind="stable")]
        hasta = np.searchsorted(t[cand], t[idx_i[m]])          # filas del grupo anteriores a la fila i
        hasta = np.maximum(hasta, 1)                            # la primera fila toma como pareja la siguiente
        desde = np.maximum(hasta - VENTANA, 0)
        j[m] = cand[np.minimum(desde + (rng.random(m.sum()) * (hasta - desde)).astype(np.int64), len(cand) - 1)]
    return j


def rasgos(i, j):
    return np.hstack([X[i], X[i] - X[j]])


def padre(params, tr, va, variante, rng, motor="XGBoost"):
    i = np.repeat(tr, PARES_POR_FILA)
    j = pareja(i, tr, variante, rng)
    m = construir(motor, params, n_jobs=14)
    m.fit(rasgos(i, j), y[i] - y[j])
    iv = np.repeat(va, ANCLAS)
    jv = pareja(iv, tr, variante, rng)
    return (y[jv] + m.predict(rasgos(iv, jv))).reshape(len(va), ANCLAS).mean(1)


# Motor de PADRE y la corrida del Entregable 3 de la que salen sus hiperparámetros y su pronóstico directo
MOTORES = {"XGBoost": ("XGBoost__Aleatoria", "XGB", "padre_exploracion.csv"),  # argumentos: motor [variantes]
           "Random Forest": ("Random_Forest__Bayesiana_Optuna", "RF", "padre_rf.csv")}

if __name__ == "__main__":
    motor = sys.argv[1] if len(sys.argv) > 1 else "XGBoost"   # python padre_exploracion.py "Random Forest"
    corrida, corto, salida = MOTORES[motor]
    variantes = sys.argv[2].split(",") if len(sys.argv) > 2 else (
        ["estacion", "vecinas"] if motor == "XGBoost" else ["estacion"])
    filas = []
    for n, tr, va in pliegues_externos(TR):
        params = json.loads((RESULTADOS / "corridas" / f"{corrida}__{n}.json")
                            .read_text(encoding="utf-8"))["mejor_params"]
        g = np.load(RESULTADOS / "corridas" / f"{corrida}__{n}.npy")
        fila = {"pliegue": n, motor: rmse(y[va], g)}
        for v in variantes:
            t0 = time.time()
            p = padre(params, tr, va, v, np.random.default_rng(SEED), motor)
            fila[f"PADRE {v}"] = rmse(y[va], p)
            np.save(RESULTADOS / "mt_oos" / f"PADRE_{v}_{corto}__{n}.npy", p)   # para combinarlo después
            fila[f"PADRE {v} + {corto}"] = rmse(y[va], (p + g) / 2)
            print(n, v, f"{time.time() - t0:.0f} s", round(fila[f"PADRE {v}"], 4), flush=True)
        print(fila, flush=True)
        filas.append(fila)
        pd.DataFrame(filas).to_csv(RESULTADOS / salida, index=False)
    R = pd.DataFrame(filas).set_index("pliegue")
    print(R.round(4).to_string())
    print(R.mean().round(4).to_string())

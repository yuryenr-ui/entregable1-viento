"""Complejidad empírica del modelo original: tiempos de entrenamiento y predicción al variar n y p.

Uso (desde la carpeta entregable1):
    python scripts/padre_complejidad.py             # PADRE-XGB y PADRE-RF, con XGBoost y Random Forest solos de referencia
    python scripts/padre_complejidad.py --hilos 10  # menos hilos para el Random Forest (PC menos cargado)

La complejidad teórica (e3/modelo_original.py) dice que PADRE entrena su motor con P·n filas y 2p columnas y que
predecir q filas cuesta M·q evaluaciones del motor. Medimos con submuestras del entrenamiento (2020–2024) y los
hiperparámetros de la búsqueda final: n ∈ {5, 10, 20, 40, 80} mil filas con las 26 predictoras, y p ∈ {6, 13, 20,
26} predictoras con 20 mil filas. Cada tiempo es la mediana de 3 repeticiones; la pendiente en escala log-log
estima el exponente. Resultado en resultados/e3/padre_complejidad.csv.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                                   # noqa: E402
from e3.datos import RESULTADOS, cargar                               # noqa: E402
from e3.modelo_original import PadreEstacionRegresor                  # noqa: E402
from e3.modelos import construir                                      # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
MOTORES = {"XGB": ("XGBoost", "XGBoost__Aleatoria__final"), "RF": ("Random Forest", "Random_Forest__Bayesiana_Optuna__final")}
N_FILAS, P_COLS, N_FIJO, Q, REPS = [5000, 10000, 20000, 40000, 80000], [6, 13, 20, 26], 20000, 5000, 3


def medir(modelo, Xa, ya, Xq):
    """Mediana de REPS tiempos de ajuste y de predicción."""
    ta, tp = [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); modelo.fit(Xa, ya); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); modelo.predict(Xq); tp.append(time.perf_counter() - t0)
    return float(np.median(ta)), float(np.median(tp))


def exponentes(R):
    """Pendiente de log(tiempo) frente a log(n) o log(p), por modelo y eje; se guarda en un CSV aparte."""
    filas = []
    for (modelo, eje), g in R.groupby(["modelo", "eje"]):
        x = np.log(g[eje])
        filas.append({"modelo": modelo, "eje": eje,
                      "exponente ajuste": np.polyfit(x, np.log(g["ajuste (s)"]), 1)[0],
                      "exponente predicción": np.polyfit(x, np.log(g["predicción de 5 000 filas (s)"]), 1)[0]})
    E = pd.DataFrame(filas).set_index(["modelo", "eje"])
    E.to_csv(RESULTADOS / "padre_complejidad_exponentes.csv")
    return E


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=14)
    args = ap.parse_args()
    TR, _, F = cargar()
    rng = np.random.default_rng(SEED)
    co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
    X = np.column_stack([TR.est.to_numpy().astype(float), TR[F].to_numpy(np.float32)])
    y = TR.y.to_numpy()
    orden_cols = rng.permutation(len(F))                                     # subconjuntos anidados de predictoras
    filas = []
    for corto, (motor, corrida) in MOTORES.items():
        params = json.loads((RESULTADOS / "corridas" / f"{corrida}.json").read_text(encoding="utf-8"))["mejor_params"]
        for eje, valores in [("n", N_FILAS), ("p", P_COLS)]:
            for v in valores:
                n, p = (v, len(F)) if eje == "n" else (N_FIJO, v)
                idx = rng.choice(len(y), n + Q, replace=False)
                cols = np.r_[0, 1 + np.sort(orden_cols[:p])]                 # estación + p predictoras
                Xa, ya, Xq = X[idx[:n]][:, cols], y[idx[:n]], X[idx[n:]][:, cols]
                base = construir(motor, params, n_jobs=args.hilos)
                padre = PadreEstacionRegresor(construir(motor, params, n_jobs=args.hilos), pares=4, anclas=20,
                                              coords=co)
                for nombre, m, Xa_, Xq_ in [(motor, base, Xa[:, 1:], Xq[:, 1:]), (f"PADRE-{corto}", padre, Xa, Xq)]:
                    ta, tp = medir(m, Xa_, ya, Xq_)
                    filas.append({"modelo": nombre, "eje": eje, "n": n, "p": p, "ajuste (s)": ta,
                                  "predicción de 5 000 filas (s)": tp})
                    print(filas[-1], flush=True)
    R = pd.DataFrame(filas)
    R.to_csv(RESULTADOS / "padre_complejidad.csv", index=False)
    print(exponentes(R).round(2).to_string())

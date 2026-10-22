"""Robustez del modelo original en el bucle externo (4 trimestres de 2024): hiperparámetros propios, muestra y ruido.

Uso (desde la carpeta entregable1, después de ``ensamble_padre.py``):
    python scripts/padre_robustez.py

Es un análisis de sensibilidad, no una selección: el modelo propuesto queda como está (4 pares, 20 anclas). Para
que sea barato usamos el motor XGBoost (GPU) y no tocamos 2025:

1. Sensibilidad a ``pares ∈ {2, 4, 8}`` y ``anclas ∈ {10, 20, 40}``: RMSE de PADRE-XGB y del ensamble de cinco
   cambiando solo ese componente (los otros cuatro salen de la caché).
2. Menor tamaño de muestra: XGBoost, PADRE-XGB y su promedio entrenados con el 10, 25, 50 y 100 % de las filas de
   entrenamiento de cada pliegue (las mismas filas para los tres).
3. Ruido: XGBoost, PADRE-XGB y su promedio con ruido gaussiano agregado al objetivo de entrenamiento
   (σ = 0, 0.5 y 1 m/s), evaluados contra el objetivo limpio.

Resultados en resultados/e3/ensamble_padre/robustez_*.csv.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
warnings.filterwarnings("ignore")
from e3 import SEED                                                                    # noqa: E402
from e3.modelo_original import PadreEstacionRegresor                                    # noqa: E402
from e3.modelos import construir                                                        # noqa: E402
from ensamble_padre import (OPT, RESULTADOS, SALIDA, TR, X, co, est, nombre, padre,     # noqa: E402
                            pliegues_externos, y)

sys.stdout.reconfigure(encoding="utf-8")
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def padre_xgb(params, Xa, ea, ya, Xe, ee, pares=4, anclas=20):
    m = PadreEstacionRegresor(construir("XGBoost", params, n_jobs=8), pares=pares, anclas=anclas, coords=co)
    return m.fit(np.column_stack([ea.astype(float), Xa]), ya).predict(np.column_stack([ee.astype(float), Xe]))


if __name__ == "__main__":
    sens, muestra, ruido = [], [], []
    for n, tr, va in pliegues_externos(TR):
        params = json.loads((RESULTADOS / "corridas" / f"{nombre('XGBoost', 'Aleatoria', n)}.json")
                            .read_text(encoding="utf-8"))["mejor_params"]
        Xe, ye, ee = X[va], y[va], est[va]
        otros = [np.load(RESULTADOS / "corridas" / f"{nombre(m, OPT[m], n)}.npy").astype(float)
                 for m in ["Random Forest", "XGBoost", "SVR"]] + [padre("RF", n, tr, Xe, ee).astype(float)]
        # 1. pares y anclas
        for P in [2, 4, 8]:
            for M in [10, 20, 40]:
                p = padre_xgb(params, X[tr], est[tr], y[tr], Xe, ee, P, M)
                sens.append({"pliegue": n, "pares": P, "anclas": M, "RMSE PADRE-XGB": rmse(ye, p),
                             "RMSE ensamble": rmse(ye, np.mean(otros + [p], 0))})
                print(sens[-1], flush=True)
        # 2. menos filas de entrenamiento (submuestra anidada: las filas del 10 % están en la del 25 %, etc.)
        orden = np.random.default_rng(SEED).permutation(tr)
        for f in [0.10, 0.25, 0.50, 1.00]:
            sub = np.sort(orden[:int(f * len(tr))])
            px = construir("XGBoost", params).fit(X[sub], y[sub]).predict(Xe)
            pp = padre_xgb(params, X[sub], est[sub], y[sub], Xe, ee)
            muestra.append({"pliegue": n, "fracción": f, "filas": len(sub), "RMSE XGBoost": rmse(ye, px),
                            "RMSE PADRE-XGB": rmse(ye, pp), "RMSE promedio": rmse(ye, (px + pp) / 2)})
            print(muestra[-1], flush=True)
        # 3. ruido en el objetivo de entrenamiento
        for s in [0.0, 0.5, 1.0]:
            yr = y[tr] + np.random.default_rng(SEED).normal(scale=s, size=len(tr))
            px = construir("XGBoost", params).fit(X[tr], yr).predict(Xe)
            pp = padre_xgb(params, X[tr], est[tr], yr, Xe, ee)
            ruido.append({"pliegue": n, "σ ruido (m/s)": s, "RMSE XGBoost": rmse(ye, px),
                          "RMSE PADRE-XGB": rmse(ye, pp), "RMSE promedio": rmse(ye, (px + pp) / 2)})
            print(ruido[-1], flush=True)
    for nom, filas, clave in [("sensibilidad", sens, ["pares", "anclas"]), ("muestra", muestra, ["fracción"]),
                              ("ruido", ruido, ["σ ruido (m/s)"])]:
        D = pd.DataFrame(filas)
        D.to_csv(SALIDA / f"robustez_{nom}.csv", index=False)
        R = D.drop(columns="pliegue").groupby(clave).agg(["mean", "std"])
        print(f"\n### {nom}\n{R.round(4).to_string()}", flush=True)

"""Corrección multi-tarea ajustada con residuos fuera de muestra (OOS) y con un promedio de modelos como base.

La primera exploración (``mt_exploracion.py``) ajustaba la corrección con los residuos del XGBoost sobre sus
propias filas de entrenamiento. Esos residuos son más pequeños que los reales (el modelo ya vio esas filas), así
que la corrección aprende un error que no es el que comete al pronosticar. Aquí, dentro del entrenamiento de cada
pliegue externo de 2024, se obtienen pronósticos fuera de muestra con validación cruzada por bloques de tiempo
contiguos (con una franja de 48 h purgada a cada lado) y la corrección se ajusta con esos residuos.

Bases comparadas: XGBoost, promedio RF+XGB y promedio RF+XGB+SVR (configuraciones elegidas en el Entregable 2
para cada pliegue). 2025 no se usa.

Etapa 1 (lenta, reanudable): guarda en ``resultados/e3/mt_oos/`` los pronósticos OOS y de validación de cada base.
Etapa 2: rejilla de la corrección y tabla resumen.
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3.datos import RESULTADOS, cargar, pliegues_externos  # noqa: E402
from e3.modelos import construir                            # noqa: E402
from e3.multitarea import CorreccionMultitarea              # noqa: E402

warnings.filterwarnings("ignore")
SALIDA = RESULTADOS / "mt_oos"
SALIDA.mkdir(exist_ok=True)
CORRIDAS = {"XGBoost": "XGBoost__Aleatoria", "Random Forest": "Random_Forest__Bayesiana_Optuna",
            "SVR": "SVR__Rejilla"}
K_BLOQUES, PURGA = 5, pd.Timedelta("48h")

TR, TE, F = cargar()
X, y, est, t = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy(), TR.t.to_numpy()
co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def bloques(tr):
    """Pares (ajuste, fuera) de la validación cruzada por bloques de tiempo contiguos dentro de ``tr``."""
    tt = t[tr]
    orden = np.argsort(tt, kind="stable")
    out = []
    for parte in np.array_split(orden, K_BLOQUES):
        fuera = np.zeros(len(tr), bool)
        fuera[parte] = True
        a, z = tt[parte].min(), tt[parte].max()
        ajuste = (tt < a - PURGA) | (tt > z + PURGA)
        out.append((tr[ajuste], tr[fuera]))
    return out


def pronosticos(nombre, pliegue, tr, va):
    """Pronóstico OOS sobre ``tr`` y pronóstico de validación (modelo con todo ``tr``), con caché en disco."""
    f = SALIDA / f"{CORRIDAS[nombre]}__{pliegue}.npz"
    if f.exists():
        d = np.load(f)
        return d["oos"], d["va"]
    params = json.loads((RESULTADOS / "corridas" / f"{CORRIDAS[nombre]}__{pliegue}.json")
                        .read_text(encoding="utf-8"))["mejor_params"]
    oos = np.full(len(tr), np.nan)
    pos = {p: i for i, p in enumerate(tr)}
    for aj, fu in bloques(tr):
        m = construir(nombre, params, n_jobs=14).fit(X[aj], y[aj])
        oos[[pos[p] for p in fu]] = m.predict(X[fu])
    p_va = construir(nombre, params, n_jobs=14).fit(X[tr], y[tr]).predict(X[va])
    np.savez(f, oos=oos, va=p_va)
    return oos, p_va


if __name__ == "__main__":
    filas = []
    for n, tr, va in pliegues_externos(TR):
        P = {}
        for nombre in CORRIDAS:
            t0 = time.time()
            P[nombre] = pronosticos(nombre, n, tr, va)
            print(n, nombre, f"{time.time() - t0:.0f} s", "OOS", round(rmse(y[tr], P[nombre][0]), 4),
                  "val", round(rmse(y[va], P[nombre][1]), 4), flush=True)
        bases = {"XGB": ["XGBoost"], "RF+XGB": ["Random Forest", "XGBoost"],
                 "RF+XGB+SVR": ["Random Forest", "XGBoost", "SVR"]}
        for b, ms in bases.items():
            g_oos = np.mean([P[m][0] for m in ms], 0)
            g_va = np.mean([P[m][1] for m in ms], 0)
            filas.append({"pliegue": n, "base": b, "residuos": "ninguna", "lam1": np.nan, "lam2": np.nan,
                          "rmse": rmse(y[va], g_va), "n": len(va)})
            for lam1, lam2 in itertools.product([100, 1000, 10000, 1e5], [0, 100, 1000, 10000, 1e7]):
                c = CorreccionMultitarea(co.index, co.lat, co.lon, lam1, lam2, 4, 50).fit(
                    X[tr], g_oos, y[tr], est[tr])
                filas.append({"pliegue": n, "base": b, "residuos": "OOS", "lam1": lam1, "lam2": lam2,
                              "rmse": rmse(y[va], c.predict(X[va], g_va, est[va])), "n": len(va)})
        pd.DataFrame(filas).to_csv(RESULTADOS / "mt_oos.csv", index=False)

    R = pd.DataFrame(filas)
    ref = R[(R.base == "XGB") & (R.residuos == "ninguna")].set_index("pliegue").rmse
    R["mejora %"] = 100 * (1 - R.rmse / R.pliegue.map(ref))
    t_ = R.groupby(["base", "residuos", "lam1", "lam2"], dropna=False)[["rmse", "mejora %"]].mean()
    print(t_.sort_values("rmse").round(4).head(20).to_string())
    print(R[R.residuos == "ninguna"].groupby("base")[["rmse", "mejora %"]].mean().round(4).to_string())

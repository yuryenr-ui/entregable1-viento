"""Combinaciones del promedio de modelos, la corrección multi-tarea y PADRE en los 4 trimestres de 2024.

Usa los pronósticos guardados por ``mt_oos.py`` y ``padre_exploracion.py`` (motores XGBoost y Random Forest).
Para cada combinación reporta el RMSE por trimestre, el RMSE con todas las filas de 2024 y la prueba de
Diebold-Mariano contra XGBoost (el modelo elegido en el Entregable 3) y contra Random Forest: diferencia de
pérdidas cuadráticas promediada por hora de emisión entre estaciones, con varianza de Newey-West a 24 rezagos.
La configuración de la corrección es fija (λ₁ = λ₂ = 1000) y 2025 no se usa.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_oos as M  # noqa: E402

LAM1, LAM2 = 1000, 1000


def dm(e_ref, e_mod, t, rezagos=24):
    """Estadístico DM (positivo = el modelo tiene menor error que la referencia) y su p-valor bilateral."""
    d = pd.Series(e_ref ** 2 - e_mod ** 2).groupby(t).mean().sort_index().to_numpy()
    n, u = len(d), d - d.mean()
    v = u @ u / n
    for k in range(1, rezagos + 1):
        v += 2 * (1 - k / (rezagos + 1)) * (u[k:] @ u[:-k]) / n
    s = d.mean() / np.sqrt(v / n)
    return s, 2 * stats.norm.sf(abs(s))


if __name__ == "__main__":
    E, T, filas = {}, [], []
    for n, tr, va in M.pliegues_externos(M.TR):
        P = {m: np.load(M.SALIDA / f"{M.CORRIDAS[m]}__{n}.npz") for m in M.CORRIDAS}
        go = np.mean([P[m]["oos"] for m in P], 0)
        gv = np.mean([P[m]["va"] for m in P], 0)
        mt = M.CorreccionMultitarea(M.co.index, M.co.lat, M.co.lon, LAM1, LAM2, 4, 50).fit(
            M.X[tr], go, M.y[tr], M.est[tr]).predict(M.X[va], gv, M.est[va])
        px = np.load(M.SALIDA / f"PADRE_estacion_XGB__{n}.npy")
        pr = np.load(M.SALIDA / f"PADRE_estacion_RF__{n}.npy")
        pred = {
            "XGBoost": P["XGBoost"]["va"],
            "Random Forest": P["Random Forest"]["va"],
            "Promedio RF+XGB+SVR": gv,
            "Promedio + MT": mt,
            "PADRE-XGB + XGB": (px + P["XGBoost"]["va"]) / 2,
            "PADRE-RF + RF": (pr + P["Random Forest"]["va"]) / 2,
            "PADRE-XGB + PADRE-RF": (px + pr) / 2,
            "Promedio 5 (RF, XGB, SVR, 2 PADRE)": (P["Random Forest"]["va"] + P["XGBoost"]["va"]
                                                   + P["SVR"]["va"] + px + pr) / 5,
            "(Promedio + MT) + 2 PADRE": (mt + px + pr) / 3,
        }
        yv = M.y[va]
        for k, p in pred.items():
            E.setdefault(k, []).append(yv - p)
            filas.append({"pliegue": n, "modelo": k, "rmse": M.rmse(yv, p)})
        T.append(M.t[va])

    T = np.concatenate(T)
    E = {k: np.concatenate(v) for k, v in E.items()}
    R = pd.DataFrame(filas).pivot(index="modelo", columns="pliegue", values="rmse")
    R["media trimestres"] = R.mean(1)
    R["RMSE 2024"] = [float(np.sqrt(np.mean(E[k] ** 2))) for k in R.index]
    for ref in ["XGBoost", "Random Forest"]:
        r = [dm(E[ref], E[k], T) for k in R.index]
        R[f"DM vs {ref}"] = [s for s, _ in r]
        R[f"p vs {ref}"] = [p for _, p in r]
    R = R.sort_values("RMSE 2024")
    R.to_csv(M.RESULTADOS / "combinacion_exploracion.csv")
    pd.set_option("display.width", 250)
    print(R.round(4).to_string())

"""Sensibilidad a la semilla del ensamble PADRE (5) en 2025.

Uso (desde la carpeta entregable1, después de ``ensamble_padre.py``):
    python scripts/ensamble_semillas.py

Los dos PADRE eligen parejas y anclas al azar y sus motores también tienen componentes aleatorias. Se
reentrenan PADRE-XGB y PADRE-RF con otras cuatro semillas, con la misma configuración, y se repiten las pruebas
frente a los 7 modelos (DM-HLN con Holm) y Clark-West frente a Random Forest. Resultado en
resultados/e3/ensamble_padre/semillas.csv (los PADRE de cada semilla quedan en caché).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                          # noqa: E402
from e3 import evaluacion as ev                              # noqa: E402
from e3.datos import RESULTADOS, cargar                      # noqa: E402
from e3.padre import PadreEstacion                           # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RESULTADOS / "ensamble_padre"
MOTORES = {"XGB": ("XGBoost", "XGBoost__Aleatoria"), "RF": ("Random Forest", "Random_Forest__Bayesiana_Optuna")}

TR, TE, F = cargar()
X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
X_te, y_te, est_te = TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy()
co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)
pred = {m: np.load(RESULTADOS / "corridas" / (f"{m}__{o}__final".replace(" ", "_").replace("(", "").replace(")", "")
                                             + ".npy")).astype(float) for m, o in OPT.items()}


def padre(corto, semilla):
    f = SALIDA / ("PADRE_{}__final.npy".format(corto) if semilla == SEED else f"PADRE_{corto}__final_s{semilla}.npy")
    if f.exists():
        return np.load(f)
    motor, corrida = MOTORES[corto]
    params = json.loads((RESULTADOS / "corridas" / f"{corrida}__final.json").read_text(encoding="utf-8"))["mejor_params"]
    t0 = time.time()
    p = PadreEstacion(params, co, pares=4, anclas=20, motor=motor, n_jobs=14, semilla=semilla).fit(X, y, est).predict(
        X_te, est_te)
    np.save(f, p)
    print(f"PADRE-{corto} semilla {semilla}: {time.time() - t0:.0f} s", flush=True)
    return p


filas = []
for s in [SEED, 1, 2, 3, 4]:
    ens = np.mean([pred["Random Forest"], pred["XGBoost"], pred["SVR"], padre("XGB", s), padre("RF", s)], 0)
    L = ev.perdida_diaria(TE.t, y_te, {"ens": ens, **pred})
    p_dm = {m: ev.diebold_mariano(L.ens, L[m])["p_valor"] for m in OPT}
    holm = dict(zip(p_dm, multipletests(list(p_dm.values()), method="holm")[1]))
    dp = ev.perdida_diaria(TE.t, pred["Random Forest"], {"d": ens}).d
    filas.append({"semilla": s, "RMSE ensamble": float(np.sqrt(np.mean((y_te - ens) ** 2))),
                  "DM vs RF": ev.diebold_mariano(L.ens, L["Random Forest"])["DM_HLN"], "p Holm vs RF": holm["Random Forest"],
                  "p Holm máximo (7 rivales)": max(holm.values()),
                  "CW vs RF": ev.clark_west(L["Random Forest"], L.ens, dp)["CW"]})
    print(filas[-1], flush=True)
R = pd.DataFrame(filas).set_index("semilla")
R.loc["media"] = R.mean()
R.loc["desv. est."] = R.iloc[:-1].std()
R.to_csv(SALIDA / "semillas.csv")
print(R.round(4).to_string())

"""Análisis complementarios del modelo propuesto (PADRE por estación + XGBoost) con las pruebas del Entregable 3.

Uso (desde la carpeta entregable1, después de ``padre_anidado.py --final``):
    python scripts/padre_analisis.py

Partes (cada una guarda su tabla en resultados/e3/padre_analisis/; las semillas quedan en caché):
1. Sensibilidad a la semilla: el modelo final se reentrena con otras semillas (anclas y parejas al azar,
   submuestreo de XGBoost) con la misma configuración y el mismo peso w, y se evalúa en 2025.
2. Clark-West frente a XGBoost: el modelo propuesto contiene a XGBoost (w = 0), así que es un modelo anidado.
3. Tamaño del efecto (δ de Cliff y d de Cohen) frente a los 7 modelos, sobre la pérdida diaria de 2025.
4. Residuos en 2025: White, BDS y Ljung-Box por estación, junto a XGBoost y Random Forest.
5. Friedman y Nemenyi en 2024 con bloques estación × trimestre, los 7 modelos más el propuesto.
6. Costo computacional frente a los modelos del curso (búsqueda, ajuste final y predicción).
7. Importancia por permutación en 2025 (aumento del RMSE al desordenar cada predictora).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                                     # noqa: E402
from e3 import evaluacion as ev                                         # noqa: E402
from e3.datos import RESULTADOS, cargar, pliegues_externos              # noqa: E402
from e3.modelos import ESPACIOS, construir                              # noqa: E402
from e3.padre import PadreEstacion                                      # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RESULTADOS / "padre_analisis"
SALIDA.mkdir(exist_ok=True)
ANIDADO = RESULTADOS / "padre_anidado"
PROPUESTO = "PADRE por estación + XGBoost"
SEMILLAS = [42, 1, 2, 3, 4]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

TR, TE, F = cargar()
X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
X_te, y_te, est_te = TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy()
co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
fin = json.loads((ANIDADO / "PADRE_XGB__final.json").read_text(encoding="utf-8"))
W, MEJOR = fin["peso_padre"], fin["mejor_params"]
tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)


def archivo(modelo, optimizador, pliegue):
    nombre = f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")
    return RESULTADOS / "corridas" / nombre


def guardar(df, nombre):
    df.to_csv(SALIDA / f"{nombre}.csv")
    print(f"\n### {nombre}\n{df.round(4).to_string()}", flush=True)


pred = {m: np.load(f"{archivo(m, o, 'final')}.npy").astype(float) for m, o in OPT.items()}
d_fin = np.load(ANIDADO / "PADRE_XGB__final.npz")
pred = {PROPUESTO: d_fin["mezcla"].astype(float), **pred}
L = ev.perdida_diaria(TE.t, y_te, pred)

# ---------------------------------------------------------------- 1. semillas
p_xgb = d_fin["xgb"].astype(float)
filas, modelo_42 = [], None
for s in SEMILLAS:
    f = SALIDA / f"semilla_{s}.npy"
    t0 = time.time()
    if f.exists() and s != 42:
        p_padre = np.load(f)
    else:
        m = PadreEstacion({k: MEJOR[k] for k in ESPACIOS["XGBoost"]}, co, MEJOR["pares"], MEJOR["anclas"],
                          semilla=s).fit(X, y, est)
        p_padre = m.predict(X_te, est_te)
        np.save(f, p_padre)
        if s == 42:
            modelo_42 = m
    mezcla = W * p_padre + (1 - W) * p_xgb
    Ls = ev.perdida_diaria(TE.t, y_te, {"m": mezcla, "RF": pred["Random Forest"], "XGB": pred["XGBoost"]})
    filas.append({"semilla": s, "RMSE PADRE": rmse(y_te, p_padre), "RMSE mezcla": rmse(y_te, mezcla),
                  "DM vs RF": ev.diebold_mariano(Ls.m, Ls.RF)["DM_HLN"],
                  "p vs RF": ev.diebold_mariano(Ls.m, Ls.RF)["p_valor"],
                  "DM vs XGB": ev.diebold_mariano(Ls.m, Ls.XGB)["DM_HLN"],
                  "p vs XGB": ev.diebold_mariano(Ls.m, Ls.XGB)["p_valor"], "segundos": time.time() - t0})
    print("semilla", s, round(filas[-1]["RMSE mezcla"], 4), flush=True)
sem = pd.DataFrame(filas).set_index("semilla")
sem.loc["media"] = sem.mean()
sem.loc["desv. est."] = sem.iloc[:-1].std()
guardar(sem, "1_semillas")

# ---------------------------------------------------------------- 2. Clark-West
dp = ev.perdida_diaria(TE.t, pred["XGBoost"], {"d": pred[PROPUESTO]}).d   # media diaria de (ŷ_XGB − ŷ_propuesto)²
cw = ev.clark_west(L["XGBoost"], L[PROPUESTO], dp)
guardar(pd.DataFrame([cw], index=["XGBoost (restringido) vs propuesto (general)"]), "2_clark_west")

# ---------------------------------------------------------------- 3. tamaño del efecto
filas = []
for m in OPT:
    cl = ev.delta_cliff(L[PROPUESTO], L[m])
    filas.append({"rival": m, "δ de Cliff": cl, "magnitud": ev.magnitud_cliff(cl),
                  "d de Cohen": ev.d_cohen(L[PROPUESTO], L[m])})
guardar(pd.DataFrame(filas).set_index("rival"), "3_efecto")

# ---------------------------------------------------------------- 4. residuos
filas = []
for m in [PROPUESTO, "XGBoost", "Random Forest"]:
    e = y_te - pred[m]
    wh = ev.prueba_white(e, X_te)
    ind = ev.pruebas_independencia(est_te, TE.t.to_numpy(), e)
    filas.append({"modelo": m, "media residuo": e.mean(), "asimetría": pd.Series(e).skew(),
                  "curtosis": pd.Series(e).kurt(), "White LM": wh["estadistico_LM"], "p White": wh["p_valor"],
                  "estaciones": len(ind), "% rechaza BDS": 100 * (ind.p_BDS < 0.05).mean(),
                  "% rechaza Ljung-Box": 100 * (ind.p_LjungBox < 0.05).mean()})
guardar(pd.DataFrame(filas).set_index("modelo"), "4_residuos")

# ---------------------------------------------------------------- 5. Friedman y Nemenyi (2024)
bloques = []
for n, _, va in pliegues_externos(TR):
    d = pd.DataFrame({"est": est[va], "y": y[va]})
    d[PROPUESTO] = (d.y - np.load(ANIDADO / f"PADRE_XGB__{n}.npz")["mezcla"]) ** 2
    for m, o in OPT.items():
        d[m] = (d.y - np.load(f"{archivo(m, o, n)}.npy")) ** 2
    g = d.groupby("est")
    cols = [PROPUESTO, *OPT]
    b = np.sqrt(g[cols].mean())[g.size() >= 48]
    b.index = [f"{e} · {n}" for e in b.index]
    bloques.append(b)
fr = ev.friedman_nemenyi(pd.concat(bloques))
print(f"\nFriedman: χ² = {fr['chi2']:.1f}, gl = {fr['gl']}, p = {fr['p_valor']:.2g}, N = {fr['N']}, "
      f"CD = {fr['CD']:.2f}")
rangos = fr["rangos"].to_frame("rango promedio")
rangos["Nemenyi p vs propuesto"] = fr["nemenyi"].loc[rangos.index, PROPUESTO]
rangos.loc["(Friedman)"] = [fr["chi2"], fr["p_valor"]]
guardar(rangos, "5_friedman_nemenyi")

# ---------------------------------------------------------------- 6. costo computacional
filas = [{"modelo": PROPUESTO, "optimizador": "Aleatoria", "búsqueda (min)": fin["segundos_busqueda"] / 60,
          "ajuste final (s)": fin["segundos_ajuste_final"], "predicción (s)": fin["segundos_prediccion"]}]
for m, o in OPT.items():
    r = json.loads(Path(f"{archivo(m, o, 'final')}.json").read_text(encoding="utf-8"))
    filas.append({"modelo": m, "optimizador": o, "búsqueda (min)": r["segundos_busqueda"] / 60,
                  "ajuste final (s)": r["segundos_ajuste_final"], "predicción (s)": r["segundos_prediccion"]})
guardar(pd.DataFrame(filas).set_index("modelo"), "6_costo")

# ---------------------------------------------------------------- 7. importancia por permutación
if modelo_42 is None:
    modelo_42 = PadreEstacion({k: MEJOR[k] for k in ESPACIOS["XGBoost"]}, co, MEJOR["pares"], MEJOR["anclas"]
                              ).fit(X, y, est)
xgb_fin = json.loads((RESULTADOS / "corridas" / "XGBoost__Aleatoria__final.json").read_text(encoding="utf-8"))
m_xgb = construir("XGBoost", xgb_fin["mejor_params"], n_jobs=8).fit(X, y)
idx = np.sort(np.random.default_rng(SEED).choice(len(y_te), 20000, replace=False))
Xs, ys, es = X_te[idx], y_te[idx], est_te[idx]


def pronosticos(Xp):
    px = m_xgb.predict(Xp)
    return W * modelo_42.predict(Xp, es) + (1 - W) * px, px


base_m, base_x = (rmse(ys, p) for p in pronosticos(Xs))
rng, filas = np.random.default_rng(SEED), []
for j, nombre in enumerate(F):
    Xp = Xs.copy()
    Xp[:, j] = rng.permutation(Xp[:, j])
    pm, px = pronosticos(Xp)
    filas.append({"predictora": nombre, f"Δ RMSE {PROPUESTO}": rmse(ys, pm) - base_m,
                  "Δ RMSE XGBoost": rmse(ys, px) - base_x})
imp = pd.DataFrame(filas).set_index("predictora").sort_values(f"Δ RMSE {PROPUESTO}", ascending=False)
guardar(imp, "7_importancia")

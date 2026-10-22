"""Extensión declarada del modelo propuesto: ensamble de 5 (RF, XGBoost, SVR, PADRE-XGB y PADRE-RF).

Uso (desde la carpeta entregable1):
    python scripts/ensamble_padre.py

La extensión se anunció junto con el modelo principal, antes de evaluar 2025, con la configuración de la
exploración de 2024: PADRE por estación con 4 parejas por fila y 20 anclas, cuyos motores (XGBoost y Random
Forest) usan los hiperparámetros que eligió el Entregable 2 en cada pliegue, y un promedio simple de los cinco
pronósticos, sin pesos ajustados. Los tres modelos del curso entran con sus pronósticos guardados.

Se evalúa en los 4 trimestres de 2024 (bucle externo) y una sola vez en 2025, con las pruebas del Entregable 2:
DM-HLN con Holm frente a los 7 modelos, Clark-West frente a Random Forest y XGBoost (el ensamble contiene a
ambos), MCS y SPA. Resultados en resultados/e3/ensamble_padre/ (los PADRE quedan en caché).
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
from e3 import evaluacion as ev                                       # noqa: E402
from e3.datos import RESULTADOS, cargar, pliegues_externos           # noqa: E402
from e3.padre import PadreEstacion                                   # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RESULTADOS / "ensamble_padre"
SALIDA.mkdir(exist_ok=True)
ENSAMBLE = "Ensamble PADRE (5)"
MOTORES = {"XGB": ("XGBoost", "XGBoost__Aleatoria"), "RF": ("Random Forest", "Random_Forest__Bayesiana_Optuna")}
DIRECTOS = {"Random Forest": "Random_Forest__Bayesiana_Optuna", "XGBoost": "XGBoost__Aleatoria", "SVR": "SVR__Rejilla"}

TR, TE, F = cargar()
X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)


def nombre(modelo, optimizador, pliegue):
    return f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")


def padre(corto, pliegue, tr, X_ev, est_ev):
    f = SALIDA / f"PADRE_{corto}__{pliegue}.npy"
    if f.exists():
        return np.load(f)
    motor, corrida = MOTORES[corto]
    params = json.loads((RESULTADOS / "corridas" / f"{corrida}__{pliegue}.json").read_text(encoding="utf-8"))["mejor_params"]
    t0 = time.time()
    p = PadreEstacion(params, co, pares=4, anclas=20, motor=motor, n_jobs=14).fit(X[tr], y[tr], est[tr]).predict(X_ev, est_ev)
    np.save(f, p)
    print(f"PADRE-{corto} {pliegue}: {time.time() - t0:.0f} s", flush=True)
    return p


def comparar(periodo, t, y_ev, preds):
    met = pd.DataFrame({k: ev.metricas(y_ev, p) for k, p in preds.items()}).T.sort_values("RMSE")
    met["reducción RMSE del ensamble (%)"] = 100 * (1 - met.loc[ENSAMBLE, "RMSE"] / met.RMSE)
    L = ev.perdida_diaria(t, y_ev, preds)
    filas = []
    for m in OPT:
        dm, gw = ev.diebold_mariano(L[ENSAMBLE], L[m]), ev.giacomini_white(L[ENSAMBLE], L[m])
        lo, hi = ev.ic_bootstrap_estacionario(L[ENSAMBLE] - L[m])
        filas.append({"rival": m, "dif. media pérdida": (L[ENSAMBLE] - L[m]).mean(), "IC 95 %": f"{lo:.4f} – {hi:.4f}",
                      "DM-HLN": dm["DM_HLN"], "p DM": dm["p_valor"], "p GW": gw["p_valor"],
                      "δ de Cliff": ev.delta_cliff(L[ENSAMBLE], L[m]), "d de Cohen": ev.d_cohen(L[ENSAMBLE], L[m])})
    cmp = pd.DataFrame(filas).set_index("rival")
    cmp["p DM (Holm)"] = multipletests(cmp["p DM"], method="holm")[1]
    cw = {}
    for ref in ["Random Forest", "XGBoost"]:          # modelos anidados en el ensamble
        dp = ev.perdida_diaria(t, preds[ref], {"d": preds[ENSAMBLE]}).d
        cw[ref] = ev.clark_west(L[ref], L[ENSAMBLE], dp)
    cw = pd.DataFrame(cw).T
    cols = [ENSAMBLE, *OPT]
    spa = pd.DataFrame({ref: ev.spa(L[ref], L[[c for c in cols if c != ref]])
                        for ref in ["Random Forest", "XGBoost", ENSAMBLE]}).T
    mcs = ev.mcs(L[cols])
    for nom, df in [("metricas", met), ("pruebas", cmp), ("clark_west", cw), ("spa", spa), ("mcs", mcs)]:
        df.to_csv(SALIDA / f"{periodo}_{nom}.csv")
        print(f"\n### {periodo} · {nom}\n{df.round(4).to_string()}", flush=True)


if __name__ == "__main__":
    for periodo, partes in [("2024", [(n, tr, va) for n, tr, va in pliegues_externos(TR)]),
                            ("2025", [("final", np.arange(len(TR)), None)])]:
        ys, ts, preds = [], [], {}
        for n, tr, va in partes:
            X_ev, y_ev, est_ev, t_ev = ((TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy(), TE.t.to_numpy())
                                        if va is None else (X[va], y[va], est[va], TR.t.to_numpy()[va]))
            p = {m: np.load(RESULTADOS / "corridas" / f"{nombre(m, o, n)}.npy").astype(float) for m, o in OPT.items()}
            p["PADRE-XGB"] = padre("XGB", n, tr, X_ev, est_ev)
            p["PADRE-RF"] = padre("RF", n, tr, X_ev, est_ev)
            p[ENSAMBLE] = np.mean([p[k] for k in ["Random Forest", "XGBoost", "SVR", "PADRE-XGB", "PADRE-RF"]], 0)
            ys.append(y_ev), ts.append(t_ev)
            for k, v in p.items():
                preds.setdefault(k, []).append(v)
            print(f"{n}: ensamble {np.sqrt(np.mean((y_ev - p[ENSAMBLE]) ** 2)):.4f} | RF "
                  f"{np.sqrt(np.mean((y_ev - p['Random Forest']) ** 2)):.4f}", flush=True)
        preds = {k: np.concatenate(v) for k, v in preds.items() if not k.startswith("PADRE-")}
        comparar(periodo, np.concatenate(ts), np.concatenate(ys), {ENSAMBLE: preds.pop(ENSAMBLE), **preds})

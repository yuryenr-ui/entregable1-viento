"""El ensamble de 5 con cada uno de los cuatro optimizadores, como la comparación 7 × 4 de los modelos del curso.

Uso (desde la carpeta entregable1):
    python scripts/ensamble_optimizadores.py            # 4 optimizadores, 2024 y 2025
    python scripts/ensamble_optimizadores.py --prueba   # solo 2024-T1 con Rejilla, para comprobar que corre
    python scripts/ensamble_optimizadores.py --hilos 10 # menos hilos para el Random Forest (PC menos cargado)

En el modelo propuesto cada componente del curso usa el optimizador que eligió su propio bucle externo (Random
Forest: Optuna; XGBoost: aleatoria; SVR: rejilla). Aquí armamos el ensamble cuatro veces más, cada una con un
solo optimizador para los cinco componentes: Random Forest, XGBoost y SVR con la búsqueda de ese optimizador
(guardada en resultados/e3/corridas/) y PADRE-XGB y PADRE-RF con los hiperparámetros de ese XGBoost y ese Random
Forest (4 pares, 20 anclas). Todo lo demás es igual al ensamble de scripts/ensamble_padre.py.

Es un análisis de sensibilidad declarado antes de correrlo (20/10/2026), no una nueva selección: el modelo
propuesto no cambia. Como en la tabla maestra de los modelos del curso, reportamos 2024 y 2025 para las cuatro
versiones, comparamos los optimizadores en 2024 con Friedman (bloques = días) y, para cada versión, contamos a
cuántos de los 7 modelos del curso supera en 2025 con Diebold-Mariano y Holm. Resultados en
resultados/e3/ensamble_optimizadores/ (los PADRE quedan en caché; los que coinciden con el ensamble original se
reutilizan).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import evaluacion as ev                                       # noqa: E402
from e3.datos import RESULTADOS, cargar, pliegues_externos           # noqa: E402
from e3.padre import PadreEstacion                                   # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RESULTADOS / "ensamble_optimizadores"
ORIGINAL = RESULTADOS / "ensamble_padre"
OPTIMIZADORES = ["Rejilla", "Aleatoria", "Bayesiana (Optuna)", "Genética (DEAP)"]
MOTOR = {"XGB": "XGBoost", "RF": "Random Forest"}
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def nombre(modelo, optimizador, pliegue):
    return f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")


def padre(corto, opt, pliegue, tr, X_ev, est_ev):
    """PADRE con el motor ``corto`` y los hiperparámetros que eligió ``opt`` en ``pliegue`` (con caché)."""
    params = json.loads((RESULTADOS / "corridas" / f"{nombre(MOTOR[corto], opt, pliegue)}.json")
                        .read_text(encoding="utf-8"))["mejor_params"]
    original = {"XGB": "Aleatoria", "RF": "Bayesiana (Optuna)"}[corto]   # optimizador del ensamble original
    if opt == original and (ORIGINAL / f"PADRE_{corto}__{pliegue}.npy").exists():
        return np.load(ORIGINAL / f"PADRE_{corto}__{pliegue}.npy")
    f = SALIDA / f"PADRE_{corto}__{nombre('', opt, pliegue).strip('_')}.npy"
    if f.exists():
        return np.load(f)
    t0 = time.time()
    p = PadreEstacion(params, co, pares=4, anclas=20, motor=MOTOR[corto], n_jobs=HILOS).fit(
        X[tr], y[tr], est[tr]).predict(X_ev, est_ev)
    np.save(f, p)
    print(f"  PADRE-{corto} {opt} {pliegue}: {time.time() - t0:.0f} s", flush=True)
    return p


def ensamble(opt, partes):
    ys, ts, comp = [], [], {}
    for n, tr, X_ev, y_ev, est_ev, t_ev in partes:
        for m in ["Random Forest", "XGBoost", "SVR"]:
            comp.setdefault(m, []).append(np.load(RESULTADOS / "corridas" / f"{nombre(m, opt, n)}.npy").astype(float))
        for c in MOTOR:
            comp.setdefault(f"PADRE-{c}", []).append(padre(c, opt, n, tr, X_ev, est_ev).astype(float))
        ys.append(y_ev), ts.append(t_ev)
    comp = {k: np.concatenate(v) for k, v in comp.items()}
    return np.concatenate(ys), np.concatenate(ts), np.mean(list(comp.values()), 0), comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--hilos", type=int, default=14, help="hilos de CPU para los motores de PADRE")
    args = ap.parse_args()
    HILOS = args.hilos
    SALIDA.mkdir(exist_ok=True)
    TR, TE, F = cargar()
    X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
    co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
    tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
    ELEGIDO = dict(tm[tm.elegido].set_index("modelo").optimizador)       # rivales: los 7 con su optimizador elegido
    ext = pliegues_externos(TR)[:1] if args.prueba else pliegues_externos(TR)
    p24 = [(n, tr, X[va], y[va], est[va], TR.t.to_numpy()[va]) for n, tr, va in ext]
    p25 = [("final", np.arange(len(TR)), TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy(),
            TE.t.to_numpy())]
    opts = ["Rejilla"] if args.prueba else OPTIMIZADORES

    filas, perd24 = [], {}
    for o in opts:
        print(o, flush=True)
        y24, t24, e24, c24 = ensamble(o, p24)
        fila = {"optimizador": o, "RMSE 2024": rmse(y24, e24),
                **{f"RMSE 2024 {k}": rmse(y24, v) for k, v in c24.items()}}
        perd24[o] = ev.perdida_diaria(t24, y24, {o: e24})[o]
        if not args.prueba:
            y25, t25, e25, c25 = ensamble(o, p25)
            rivales = {m: np.load(RESULTADOS / "corridas" / f"{nombre(m, op, 'final')}.npy").astype(float)
                       for m, op in ELEGIDO.items()}
            L = ev.perdida_diaria(t25, y25, {"ens": e25, **rivales})
            dm = pd.DataFrame({m: ev.diebold_mariano(L.ens, L[m]) for m in ELEGIDO}).T
            dm["p Holm"] = multipletests(dm.p_valor, method="holm")[1]
            dm.to_csv(SALIDA / f"2025_pruebas_{nombre('', o, '').strip('_')}.csv")
            gana = (dm.diferencia_media < 0) & (dm["p Holm"] < .05)
            fila.update({"RMSE 2025": rmse(y25, e25), "MAE 2025": float(np.mean(np.abs(y25 - e25))),
                         "R2 2025": float(1 - np.sum((y25 - e25) ** 2) / np.sum((y25 - y25.mean()) ** 2)),
                         "supera con Holm (de 7)": int(gana.sum()), "p Holm vs RF": float(dm.loc["Random Forest", "p Holm"]),
                         **{f"RMSE 2025 {k}": rmse(y25, v) for k, v in c25.items()}})
        filas.append(fila)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()
               if "PADRE" not in k and k[-3:] not in ("RF", "XGB")}, flush=True)

    R = pd.DataFrame(filas).set_index("optimizador")
    if not args.prueba:
        P = pd.DataFrame(perd24)
        fr = stats.friedmanchisquare(*[P[o] for o in P])
        R["rango medio 2024 (Friedman)"] = P.rank(axis=1).mean()
        print(f"\nFriedman 2024 (bloques = {len(P)} días): χ² = {fr.statistic:.2f}, p = {fr.pvalue:.3g}")
        (SALIDA / "friedman.json").write_text(json.dumps({"chi2": float(fr.statistic), "p": float(fr.pvalue),
                                                          "dias": len(P)}), encoding="utf-8")
    R.to_csv(SALIDA / ("resumen_prueba.csv" if args.prueba else "resumen.csv"))
    print(R[[c for c in R.columns if "PADRE" not in c and not c.startswith(("RMSE 2024 ", "RMSE 2025 "))]]
          .round(4).to_string())

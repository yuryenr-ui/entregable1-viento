"""Validación anidada del modelo propuesto: PADRE por estación + XGBoost, con el mismo esquema del Entregable 2.

Uso (desde la carpeta entregable1):
    python scripts/padre_anidado.py             # 4 trimestres de 2024 (bucle externo)
    python scripts/padre_anidado.py --final     # además, búsqueda con 2024 y evaluación única en 2025
    python scripts/padre_anidado.py --prueba    # versión mínima para comprobar que todo corre
    python scripts/padre_anidado.py --anclas 10 20 40 80   # espacio ampliado (carpeta padre_anidado_a80)
    python scripts/padre_anidado.py --optimizador rejilla  # otro optimizador (carpeta padre_anidado_rejilla)

Para cada pliegue externo, igual que los 7 modelos del curso:
- bucle interno con las mismas submuestras (40 mil filas de ajuste y 10 mil de validación por trimestre);
- búsqueda de 40 configuraciones (aleatoria por defecto; ``--optimizador`` elige rejilla, bayesiana o
  genética, con el mismo presupuesto y las mismas implementaciones de e3/optimizadores.py) sobre el espacio
  del XGBoost más los hiperparámetros de PADRE (parejas por fila y anclas). Para cada configuración se elige también el peso ``w`` de la mezcla
  ``w · PADRE + (1 − w) · XGBoost``; el XGBoost de la mezcla usa la configuración que eligió su propia búsqueda
  en ese pliegue (Entregable 2), ajustado sobre las mismas submuestras;
- la configuración y el peso elegidos se reentrenan con todo el entrenamiento del pliegue y se miden una vez.
Cada pliegue guarda un JSON y sus pronósticos en resultados/e3/padre_anidado/; si el proceso se corta, al
relanzarlo los pliegues terminados no se repiten.
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
from e3 import SEED                                                                     # noqa: E402
from e3.datos import RESULTADOS, TEST_INI, TRIMESTRES, cargar, particion_interna, pliegues_externos  # noqa: E402
from e3.modelos import ESPACIOS, construir                                              # noqa: E402
from e3.optimizadores import OPTIMIZADORES                                                 # noqa: E402
from e3.padre import ESPACIO_PADRE, PadreEstacion, mejor_peso                           # noqa: E402

warnings.filterwarnings("ignore")
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
CORTOS = {"rejilla": "Rejilla", "aleatoria": "Aleatoria", "bayesiana": "Bayesiana (Optuna)", "genetica": "Genética (DEAP)"}


def pliegue_padre(pliegue, cfg, carpeta, TR, TE, F, co, ESPACIO, optimizar):
    salida = carpeta / f"PADRE_XGB__{pliegue}.json"
    if salida.exists():
        return json.loads(salida.read_text(encoding="utf-8"))
    X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
    if pliegue == "final":
        fin, n_int = TEST_INI, 4
        tr_ext, X_ev, y_ev, est_ev = np.arange(len(TR)), TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy()
    else:
        _, tr_ext, va_ext = next(p for p in pliegues_externos(TR) if p[0] == pliegue)
        fin, n_int = TRIMESTRES[4 + int(pliegue[-1]) - 1], 3
        X_ev, y_ev, est_ev = X[va_ext], y[va_ext], est[va_ext]
    if cfg["prueba"]:
        n_int = 1
    internos = particion_interna(TR, fin, n_int, cfg["n_ajuste"], cfg["n_valid"])
    corrida_xgb = json.loads((RESULTADOS / "corridas" / f"XGBoost__Aleatoria__{pliegue}.json").read_text(encoding="utf-8"))
    px_int = [construir("XGBoost", corrida_xgb["mejor_params"], n_jobs=8).fit(X[tr], y[tr]).predict(X[va])
              for tr, va in internos]
    y_int = [y[va] for _, va in internos]
    pesos = {}

    def objetivo(params):
        motor = {k: params[k] for k in ESPACIOS["XGBoost"]}
        pp = [PadreEstacion(motor, co, params["pares"], params["anclas"]).fit(X[tr], y[tr], est[tr])
              .predict(X[va], est[va]) for tr, va in internos]
        w, err = mejor_peso(y_int, pp, px_int)
        pesos[tuple(sorted(params.items()))] = w
        return err

    t0 = time.perf_counter()
    res = optimizar(ESPACIO, objetivo, cfg["presupuesto"], semilla=SEED)
    t_busqueda = time.perf_counter() - t0
    for h in res["historial"]:
        h["peso"] = pesos[tuple(sorted(h["params"].items()))]
    mejor = res["mejor_params"]
    w = pesos[tuple(sorted(mejor.items()))]
    t1 = time.perf_counter()
    m = PadreEstacion({k: mejor[k] for k in ESPACIOS["XGBoost"]}, co, mejor["pares"], mejor["anclas"]).fit(
        X[tr_ext], y[tr_ext], est[tr_ext])
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    p_padre = m.predict(X_ev, est_ev).astype(np.float32)
    t_pred = time.perf_counter() - t2
    p_xgb = np.load(RESULTADOS / "corridas" / f"XGBoost__Aleatoria__{pliegue}.npy")
    p_mix = (w * p_padre + (1 - w) * p_xgb).astype(np.float32)
    np.savez(carpeta / f"PADRE_XGB__{pliegue}.npz", padre=p_padre, xgb=p_xgb, mezcla=p_mix)
    res.update({"pliegue": pliegue, "peso_padre": w, "semilla": SEED, "segundos_busqueda": t_busqueda,
                "segundos_ajuste_final": t_ajuste, "segundos_prediccion": t_pred,
                "filas_ajuste_final": int(len(tr_ext)), "filas_evaluadas": int(len(y_ev)),
                "rmse_padre": rmse(y_ev, p_padre), "rmse_xgb": rmse(y_ev, p_xgb), "rmse": rmse(y_ev, p_mix),
                "mae": float(np.mean(np.abs(y_ev - p_mix))),
                "r2": float(1 - np.sum((y_ev - p_mix) ** 2) / np.sum((y_ev - y_ev.mean()) ** 2))})
    tmp = salida.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(salida)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--final", action="store_true", help="incluir la evaluación única en 2025")
    ap.add_argument("--anclas", nargs="+", type=int, default=ESPACIO_PADRE["anclas"][1],
                    help="opciones de anclas del espacio de búsqueda")
    ap.add_argument("--optimizador", default="aleatoria", choices=list(CORTOS))
    args = ap.parse_args()
    ESPACIO = {**ESPACIOS["XGBoost"], **ESPACIO_PADRE, "anclas": ("cat", args.anclas)}
    sufijo = "" if args.anclas == ESPACIO_PADRE["anclas"][1] else f"_a{max(args.anclas)}"
    if args.optimizador != "aleatoria":
        sufijo += f"_{args.optimizador}"
    optimizar = OPTIMIZADORES[CORTOS[args.optimizador]]
    cfg = ({"prueba": True, "presupuesto": 3, "n_ajuste": 3000, "n_valid": 1000} if args.prueba else
           {"prueba": False, "presupuesto": 40, "n_ajuste": 40000, "n_valid": 10000})
    carpeta = RESULTADOS / (("padre_anidado_prueba" if args.prueba else "padre_anidado") + sufijo)
    carpeta.mkdir(exist_ok=True)
    TR, TE, F = cargar()
    co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
    pliegues = ["2024-T1"] if args.prueba else ["2024-T1", "2024-T2", "2024-T3", "2024-T4"]
    if args.final:
        pliegues.append("final")
    filas = []
    for p in pliegues:
        t0 = time.time()
        r = pliegue_padre(p, cfg, carpeta, TR, TE, F, co, ESPACIO, optimizar)
        filas.append({k: r[k] for k in ["pliegue", "rmse_xgb", "rmse_padre", "rmse", "peso_padre",
                                        "mejor_rmse_interno"]})
        print(p, f"{(time.time() - t0) / 60:.1f} min", {k: (round(v, 4) if isinstance(v, float) else v)
                                                         for k, v in filas[-1].items()}, flush=True)
    R = pd.DataFrame(filas).set_index("pliegue")
    R.to_csv(carpeta.parent / f"{carpeta.name}.csv")
    print(R.round(4).to_string())

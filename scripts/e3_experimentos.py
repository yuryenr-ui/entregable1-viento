"""Corre las 28 combinaciones modelo × optimizador con validación anidada y la evaluación final en 2025.

Uso (desde la carpeta entregable1):
    python scripts/e3_experimentos.py            # experimento completo (varias horas)
    python scripts/e3_experimentos.py --prueba   # versión mínima para comprobar que todo corre
    python scripts/e3_experimentos.py --modelos Ridge Lasso   # solo algunos modelos (por sesiones)

Cada tarea (modelo, optimizador, pliegue) guarda un JSON y sus predicciones en resultados/e3/corridas/, de
modo que si el proceso se interrumpe basta con volver a lanzarlo: las tareas terminadas no se repiten. Al
final se arma la tabla maestra resultados/e3/tabla_maestra.csv.

Esquema:
- Bucle externo: los 4 trimestres de 2024 (ventana creciente). Dentro de cada uno, la búsqueda usa como
  bucle interno los 3 trimestres anteriores, con submuestras fijas (40 mil filas de ajuste y 10 mil de
  validación por pliegue). La configuración elegida se reentrena con todo el entrenamiento del pliegue
  externo y se mide en su trimestre: esa es la métrica que se reporta (media ± desviación estándar).
- Evaluación final: la búsqueda se repite con los 4 trimestres de 2024 como bucle interno, se reentrena con
  todo 2020–2024 y se predice 2025 una sola vez.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                                    # noqa: E402
from e3.datos import RESULTADOS, TEST_INI, TRIMESTRES, cargar, particion_interna, pliegues_externos  # noqa: E402
from e3.modelos import ESPACIOS, MODELOS, construir                    # noqa: E402
from e3.optimizadores import OPTIMIZADORES, eliminacion_sucesiva        # noqa: E402

N_JOBS_MODELO = {"KNN": 4, "Random Forest": 4, "XGBoost": 2}           # núcleos por ajuste; el resto usa 1


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def tarea(modelo, optimizador, pliegue, cfg, carpeta):
    """Ejecuta una búsqueda completa y guarda su resultado. Devuelve el nombre del archivo."""
    nombre = f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")
    salida = carpeta / f"{nombre}.json"
    if salida.exists():
        return salida.name
    warnings.filterwarnings("ignore")
    TR, TE, F = cargar()
    X, y = TR[F].to_numpy(np.float32), TR.y.to_numpy()
    if pliegue == "final":
        fin, n_int = TEST_INI, 4
        tr_ext, X_ev, y_ev = np.arange(len(TR)), TE[F].to_numpy(np.float32), TE.y.to_numpy()
    else:
        _, tr_ext, va_ext = next(p for p in pliegues_externos(TR) if p[0] == pliegue)
        fin, n_int = TRIMESTRES[4 + int(pliegue[-1]) - 1], 3
        X_ev, y_ev = X[va_ext], y[va_ext]
    if cfg["prueba"]:
        n_int = 1
    internos = particion_interna(TR, fin, n_int, cfg["n_ajuste"], cfg["n_valid"])
    nj = N_JOBS_MODELO.get(modelo, 1)

    def objetivo(params, fraccion=1.0):
        errores = []
        for tr, va in internos:
            tr = tr[-max(200, int(len(tr) * fraccion)):] if fraccion < 1 else tr
            m = construir(modelo, params, n_jobs=nj).fit(X[tr], y[tr])
            errores.append(rmse(y[va], m.predict(X[va])))
        return np.mean(errores)

    t0 = time.perf_counter()
    if optimizador == "Eliminación sucesiva":
        res = eliminacion_sucesiva(ESPACIOS[modelo], objetivo, n_inicial=cfg["n_sh"])
    else:
        res = OPTIMIZADORES[optimizador](ESPACIOS[modelo], objetivo, cfg["presupuesto"], semilla=SEED)
    t_busqueda = time.perf_counter() - t0
    # reentrenar la mejor configuración con todo el entrenamiento externo y medir en el periodo reservado
    t1 = time.perf_counter()
    m = construir(modelo, res["mejor_params"], n_jobs=nj).fit(X[tr_ext], y[tr_ext])
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = m.predict(X_ev).astype(np.float32)
    t_pred = time.perf_counter() - t2
    np.save(carpeta / f"{nombre}.npy", pred)
    res.update({"modelo": modelo, "optimizador": optimizador, "pliegue": pliegue, "semilla": SEED,
                "segundos_busqueda": t_busqueda, "segundos_ajuste_final": t_ajuste,
                "segundos_prediccion": t_pred, "filas_ajuste_final": int(len(tr_ext)),
                "filas_evaluadas": int(len(y_ev)), "rmse": rmse(y_ev, pred),
                "mae": float(np.mean(np.abs(y_ev - pred))),
                "r2": float(1 - np.sum((y_ev - pred) ** 2) / np.sum((y_ev - y_ev.mean()) ** 2))})
    # escritura atómica: si el proceso se corta a mitad (por ejemplo, un apagón), no queda un JSON incompleto
    # que haga creer que la tarea terminó; al relanzar, la tarea simplemente se repite
    tmp = salida.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(salida)
    return salida.name


def tabla_maestra(carpeta):
    import pandas as pd
    filas = []
    for f in sorted(carpeta.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        filas.append({k: r.get(k) for k in ["modelo", "optimizador", "pliegue", "rmse", "mae", "r2",
                                            "mejor_rmse_interno", "evaluaciones", "segundos_busqueda",
                                            "segundos_ajuste_final", "segundos_prediccion", "semilla"]}
                     | {"mejor_params": json.dumps(r["mejor_params"], ensure_ascii=False)})
    t = pd.DataFrame(filas)
    t.to_csv(carpeta.parent / ("tabla_maestra_prueba.csv" if "prueba" in carpeta.name else "tabla_maestra.csv"),
             index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--procesos", type=int, default=6)
    ap.add_argument("--modelos", nargs="+", default=MODELOS, choices=MODELOS,
                    help="correr solo estos modelos (por sesiones); lo ya guardado no se repite")
    args = ap.parse_args()
    cfg = ({"prueba": True, "presupuesto": 4, "n_ajuste": 3000, "n_valid": 1000, "n_sh": 9} if args.prueba else
           {"prueba": False, "presupuesto": 40, "n_ajuste": 40000, "n_valid": 10000, "n_sh": 27})
    carpeta = RESULTADOS / ("corridas_prueba" if args.prueba else "corridas")
    carpeta.mkdir(parents=True, exist_ok=True)
    pliegues = ["2024-T1"] + (["final"] if args.prueba else ["2024-T2", "2024-T3", "2024-T4", "final"])
    tareas = [(m, o, p) for m in args.modelos for o in OPTIMIZADORES for p in pliegues]
    tareas += [(m, "Eliminación sucesiva", "final") for m in ["Random Forest", "XGBoost"] if m in args.modelos]
    # primero las más costosas para que el reparto entre procesos quede equilibrado
    costo = {"Random Forest": 0, "KNN": 1, "XGBoost": 2, "SVR": 3, "Árbol de decisión": 4, "Lasso": 5, "Ridge": 6}
    tareas.sort(key=lambda t: (costo[t[0]], t[2] != "final"))
    print(f"{len(tareas)} tareas, {args.procesos} procesos -> {carpeta}", flush=True)
    t0 = time.time()
    for i, nombre in enumerate(Parallel(n_jobs=args.procesos, return_as="generator_unordered")(
            delayed(tarea)(*t, cfg, carpeta) for t in tareas), 1):
        print(f"[{i}/{len(tareas)}] {nombre} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(tabla_maestra(carpeta).groupby(["modelo", "optimizador"]).rmse.mean().round(4).to_string())

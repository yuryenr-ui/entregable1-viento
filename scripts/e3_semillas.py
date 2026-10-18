"""Sensibilidad a la semilla aleatoria (sección 5.6). Se corre después de scripts/e3_experimentos.py.

Uso (desde la carpeta entregable1):
    python scripts/e3_semillas.py

1. Reentrena la configuración final de cada modelo con componente aleatoria (Árbol, Random Forest, XGBoost,
   SVR con Nyström) con 5 semillas y mide el RMSE en 2025: variabilidad por la inicialización del modelo.
2. Repite la búsqueda final de XGBoost con 3 semillas para la búsqueda aleatoria, la bayesiana y la genética:
   variabilidad de la solución encontrada por el optimizador.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3.datos import RESULTADOS, TEST_INI, cargar, particion_interna  # noqa: E402
from e3.modelos import ESPACIOS, construir                           # noqa: E402
from e3.optimizadores import OPTIMIZADORES                           # noqa: E402

CORRIDAS = RESULTADOS / "corridas"
SALIDA = RESULTADOS / "semillas"
SALIDA.mkdir(parents=True, exist_ok=True)
SEMILLAS_MODELO = [0, 1, 2, 3, 4]
SEMILLAS_BUSQUEDA = [1, 2, 3]


def mejor_optimizador(modelo):
    """Optimizador con menor RMSE medio en el bucle externo (nunca se mira 2025 para elegir)."""
    t = pd.read_csv(RESULTADOS / "tabla_maestra.csv")
    t = t[(t.modelo == modelo) & (t.pliegue != "final") & (t.optimizador != "Eliminación sucesiva")]
    return t.groupby("optimizador").rmse.mean().idxmin()


def params_final(modelo, optimizador):
    f = CORRIDAS / f"{modelo}__{optimizador}__final.json".replace(" ", "_").replace("(", "").replace(")", "")
    return json.loads(f.read_text(encoding="utf-8"))["mejor_params"]


def reentrenar(modelo, params, semilla):
    warnings.filterwarnings("ignore")
    TR, TE, F = cargar()
    m = construir(modelo, params, n_jobs=4, semilla=semilla).fit(TR[F].to_numpy(np.float32), TR.y.to_numpy())
    p = m.predict(TE[F].to_numpy(np.float32))
    return {"modelo": modelo, "semilla": semilla, "rmse": float(np.sqrt(np.mean((TE.y.to_numpy() - p) ** 2)))}


def buscar(optimizador, semilla):
    warnings.filterwarnings("ignore")
    TR, TE, F = cargar()
    X, y = TR[F].to_numpy(np.float32), TR.y.to_numpy()
    internos = particion_interna(TR, TEST_INI, 4, 40000, 10000)

    def objetivo(params):
        return np.mean([np.sqrt(np.mean((y[va] - construir("XGBoost", params, n_jobs=2).fit(X[tr], y[tr])
                                         .predict(X[va])) ** 2)) for tr, va in internos])

    r = OPTIMIZADORES[optimizador](ESPACIOS["XGBoost"], objetivo, 40, semilla=semilla)
    m = construir("XGBoost", r["mejor_params"], n_jobs=4).fit(X, y)
    p = m.predict(TE[F].to_numpy(np.float32))
    return {"optimizador": optimizador, "semilla_busqueda": semilla, "mejor_rmse_interno": r["mejor_rmse_interno"],
            "rmse_2025": float(np.sqrt(np.mean((TE.y.to_numpy() - p) ** 2))),
            "params": json.dumps(r["mejor_params"]),
            "curva": json.dumps([h["rmse"] for h in r["historial"]])}


if __name__ == "__main__":
    tareas = [(m, params_final(m, mejor_optimizador(m)), s)
              for m in ["Árbol de decisión", "Random Forest", "XGBoost", "SVR"] for s in SEMILLAS_MODELO]
    filas = Parallel(n_jobs=5)(delayed(reentrenar)(*t) for t in tareas)
    pd.DataFrame(filas).to_csv(SALIDA / "semilla_modelo.csv", index=False)
    print(pd.DataFrame(filas).groupby("modelo").rmse.agg(["mean", "std", "min", "max"]))
    filas = Parallel(n_jobs=5)(delayed(buscar)(o, s) for o in ["Aleatoria", "Bayesiana (Optuna)", "Genética (DEAP)"]
                               for s in SEMILLAS_BUSQUEDA)
    pd.DataFrame(filas).to_csv(SALIDA / "semilla_busqueda.csv", index=False)
    print(pd.DataFrame(filas).drop(columns=["curva", "params"]))

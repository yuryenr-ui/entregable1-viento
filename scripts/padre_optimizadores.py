"""Compara los cuatro optimizadores en la variante PADRE + XGBoost, como la comparación 7 × 4 de los modelos del curso.

Uso (desde la carpeta entregable1), después de correr padre_anidado.py con cada optimizador:
    python scripts/padre_optimizadores.py

Cada optimizador tiene el mismo presupuesto (40 configuraciones), las mismas submuestras y el mismo esquema
anidado. Con los 4 trimestres de 2024 se comparan los pronósticos de la mezcla: RMSE y prueba de Friedman
sobre la pérdida cuadrática diaria (bloques = días), con Nemenyi como prueba posterior.

Regla fijada ANTES de ver los resultados (20/10/2026): se elige el optimizador con menor RMSE en 2024, igual
que el bucle externo de los modelos del curso. Si es la búsqueda aleatoria, su evaluación en 2025 ya existe y no
se hace nada más. Si es otro, se ejecuta su búsqueda final y su evaluación única en 2025 con
``padre_anidado.py --final --optimizador <nombre>`` y ``padre_comparacion.py 2025 <carpeta>``. La decisión queda
en resultados/e3/padre_optimizadores.json.
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import evaluacion as ev                                       # noqa: E402
from e3.datos import RESULTADOS, cargar, pliegues_externos           # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
CARPETAS = {"Rejilla": "padre_anidado_rejilla", "Aleatoria": "padre_anidado",
            "Bayesiana (Optuna)": "padre_anidado_bayesiana", "Genética (DEAP)": "padre_anidado_genetica"}
CORTO = {"Rejilla": "rejilla", "Bayesiana (Optuna)": "bayesiana", "Genética (DEAP)": "genetica"}
PLIEGUES = ["2024-T1", "2024-T2", "2024-T3", "2024-T4"]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

if __name__ == "__main__":
    TR, TE, F = cargar()
    va = np.concatenate([v for _, _, v in pliegues_externos(TR)])
    y, t = TR.y.to_numpy()[va], TR.t.to_numpy()[va]
    P, filas = {}, []
    for o, c in CARPETAS.items():
        P[o] = np.concatenate([np.load(RESULTADOS / c / f"PADRE_XGB__{n}.npz")["mezcla"] for n in PLIEGUES]).astype(float)
        js = [json.loads((RESULTADOS / c / f"PADRE_XGB__{n}.json").read_text(encoding="utf-8")) for n in PLIEGUES]
        filas.append({"optimizador": o, "RMSE 2024": rmse(y, P[o]),
                      **{f"RMSE {n}": j["rmse"] for n, j in zip(PLIEGUES, js)},
                      "w medio": np.mean([j["peso_padre"] for j in js]),
                      "búsqueda (min, media por pliegue)": np.mean([j["segundos_busqueda"] for j in js]) / 60})
    R = pd.DataFrame(filas).set_index("optimizador").sort_values("RMSE 2024")
    L = ev.perdida_diaria(t, y, P)
    fr = stats.friedmanchisquare(*[L[o] for o in CARPETAS])
    rangos = L.rank(axis=1).mean()
    R["rango medio (Friedman)"] = rangos[R.index]
    nem = sp.posthoc_nemenyi_friedman(L.to_numpy())
    nem.index = nem.columns = list(CARPETAS)
    R.to_csv(RESULTADOS / "padre_optimizadores.csv")
    nem.to_csv(RESULTADOS / "padre_optimizadores_nemenyi.csv")
    print(R.round(4).to_string())
    print(f"\nFriedman (bloques = {len(L)} días): χ² = {fr.statistic:.2f}, p = {fr.pvalue:.3g}")
    print("Nemenyi (p):\n", nem.round(3).to_string())

    elegido = R.index[0]
    decision = {"elegido": elegido, "RMSE 2024": float(R.iloc[0]["RMSE 2024"]), "friedman_chi2": float(fr.statistic),
                "friedman_p": float(fr.pvalue), "evaluar_2025": elegido != "Aleatoria"}
    (RESULTADOS / "padre_optimizadores.json").write_text(json.dumps(decision, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    print("Decisión:", decision, flush=True)
    if decision["evaluar_2025"]:
        aqui = Path(__file__).resolve().parent
        subprocess.run([sys.executable, str(aqui / "padre_anidado.py"), "--final", "--optimizador", CORTO[elegido]],
                       check=True)
        subprocess.run([sys.executable, str(aqui / "padre_comparacion.py"), "2025", CARPETAS[elegido]], check=True)

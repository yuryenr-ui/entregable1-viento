"""Primera prueba del modelo multi-tarea con los 4 trimestres de 2024 (2025 no se usa).

Para cada pliegue externo reentrena el XGBoost elegido en ese pliegue, ajusta la corrección multi-tarea sobre
sus residuos de entrenamiento y mide el RMSE en el trimestre. Compara:
- XGBoost solo;
- corrección común a todas las estaciones (λ₂ muy grande);
- corrección independiente por estación (λ₂ = 0);
- corrección con grafo geográfico (λ₂ intermedio).
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3.datos import RESULTADOS, cargar, pliegues_externos  # noqa: E402
from e3.modelos import construir                            # noqa: E402
from e3.multitarea import CorreccionMultitarea              # noqa: E402

warnings.filterwarnings("ignore")
TR, TE, F = cargar()
X, y, est = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy()
co = pd.read_csv(RESULTADOS / "estaciones.csv", dtype={"est": str}).set_index("est")
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

filas = []
for n, tr, va in pliegues_externos(TR):
    r = json.loads((RESULTADOS / "corridas" / f"XGBoost__Aleatoria__{n}.json").read_text(encoding="utf-8"))
    m = construir("XGBoost", r["mejor_params"], n_jobs=8).fit(X[tr], y[tr])
    g_tr, g_va = m.predict(X[tr]), m.predict(X[va])
    base = rmse(y[va], g_va)
    for lam1, lam2, k, esc in itertools.product([10, 100, 1000, 10000], [0, 100, 1000, 10000, 1e7], [4], [50, 150]):
        c = CorreccionMultitarea(co.index, co.lat, co.lon, lam1, lam2, k, esc).fit(X[tr], g_tr, y[tr], est[tr])
        filas.append({"pliegue": n, "lam1": lam1, "lam2": lam2, "escala": esc, "XGBoost": base,
                      "corregido": rmse(y[va], c.predict(X[va], g_va, est[va]))})
    print(n, "XGBoost", round(base, 4), flush=True)
R = pd.DataFrame(filas)
R.to_csv(RESULTADOS / "mt_exploracion.csv", index=False)
t = R.groupby(["lam1", "lam2", "escala"])[["XGBoost", "corregido"]].mean()
t["mejora %"] = 100 * (1 - t.corregido / t.XGBoost)
print(t.sort_values("corregido").round(4).head(15).to_string())
print(t.xs(1e7, level="lam2").round(4).to_string())   # corrección común (sin multi-tarea)
print(t.xs(0, level="lam2").round(4).to_string())     # corrección independiente por estación

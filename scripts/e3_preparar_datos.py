"""Guarda el conjunto de modelado del libro (limpieza causal) para los experimentos de la sección 4.

Uso (desde la carpeta entregable1):
    python scripts/e3_preparar_datos.py

Ejecuta las celdas de entregable1.py hasta la construcción del panel causal, con las mismas funciones,
semillas y predictoras, y guarda resultados/e3/modelado.parquet. Así los modelos de la sección 4 parten
exactamente de los mismos datos que el SVR de la sección 3.
"""
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventanas
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent.parent
SALIDA = AQUI / "resultados" / "e3"
SALIDA.mkdir(parents=True, exist_ok=True)

fuente = (AQUI / "entregable1.py").read_text(encoding="utf-8")
cabeceras = re.findall(r"^# %%.*$", fuente, flags=re.M)
celdas = re.split(r"^# %%.*$", fuente, flags=re.M)[1:]
os.chdir(AQUI)
ns = {"__name__": "__e3__"}
plt.show = lambda *a, **k: plt.close("all")
inicio = time.time()
for i, (cab, c) in enumerate(zip(cabeceras, celdas), 1):
    if "[markdown]" in cab:
        continue
    c = c.replace("from IPython.display import display", "display = lambda *a, **k: None")
    exec(compile(c, f"celda_{i}", "exec"), ns)
    plt.close("all")
    if "preparar_panel(DATOS" in c:  # el panel causal ya está listo
        break
print(f"celdas ejecutadas en {time.time() - inicio:.0f} s")

D, F_ESP = ns["D"], ns["F_ESP"]
cols = ["est", "t", "t_y", "y", "hora_y", "mes_y"] + F_ESP
D[cols].to_parquet(SALIDA / "modelado.parquet", index=False)
(SALIDA / "predictoras.txt").write_text("\n".join(F_ESP), encoding="utf-8")
TR, TE = ns["TR"], ns["TE"]
print(f"{len(D):,} filas ({len(TR):,} entrenamiento, {len(TE):,} prueba), {len(F_ESP)} predictoras, "
      f"{D.est.nunique()} estaciones -> {SALIDA / 'modelado.parquet'}")

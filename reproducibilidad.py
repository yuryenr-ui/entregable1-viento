# %% [markdown]
# # Reproducibilidad y bibliografía
#
# **Curso:** Machine Learning · **Profesor:** Lihki Rubio Ortega
# **Grupo:** Yuryen Rollo, Betzaida Ruiz
#
# ## 10.1 Entorno y semillas
#
# - **Código:** https://github.com/yuryenr-ui/entregable1-viento (carpeta raíz del libro). Los módulos están en
#   `e3/` (datos, modelos, optimizadores, evaluación y el modelo original en `e3/modelo_original.py`), los
#   experimentos en `scripts/`, las pruebas unitarias en `tests/` y los artefactos serializados en `artifacts/`.
# - **Versiones exactas:** `requirements.txt`. Python 3.11.
# - **Semilla global:** `42` (`e3.SEED`), usada por numpy, scikit-learn, XGBoost, Optuna, DEAP y por los sorteos
#   de pares y anclas de PADRE (`random_state` y `random_state + 1`).
# - **Equipo:** AMD Ryzen 7 8700F (8 núcleos, 16 hilos), 32 GB de RAM y GPU NVIDIA RTX 4060 Ti de 8 GB (XGBoost con
#   `device="cuda"`). Los tiempos reportados son de esta máquina.
# - **Datos:** los CSV del IDEAM en datos.gov.co {cite}`ideam2025`; los enlaces y el procesamiento están en la
#   sección 1.3 y en la portada del libro.

# %%
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from e3 import SEED
from e3.datos import RESULTADOS, cargar, pliegues_externos

print(f"Python {platform.python_version()} | semilla {SEED}")
print({p: md.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "optuna", "deap", "shap",
                                   "statsmodels", "arch", "pytest"]})

# %% [markdown]
# ## 10.2 Orden de ejecución
#
# Todos los comandos se ejecutan desde la carpeta del repositorio. Los cálculos costosos guardan sus resultados en
# `resultados/e3/`, que está versionada; los notebooks solo leen esas tablas y predicciones, así que el libro se
# puede reconstruir sin repetir las búsquedas.
#
# | Paso | Comando | Tiempo aproximado | Produce |
# |---|---|---|---|
# | 1 | `python scripts/procesar_todo.py <descargas> --causal` (desde `../ideam_viento`) | — | Panel horario con limpieza causal |
# | 2 | Ejecutar `entregable1.ipynb` | — | Capítulo 1 (datos, EDA y modelo base) |
# | 3 | `python scripts/e3_preparar_datos.py` | 2 min | Conjunto de modelado (`modelado.parquet`) |
# | 4 | `python scripts/e3_experimentos.py` | ≈ 16 h (suma de las búsquedas) | Las búsquedas anidadas (7 modelos × 4 optimizadores) y sus pronósticos |
# | 5 | `python scripts/e3_computacional.py` y `scripts/e3_semillas.py` | — | Sección 5.3 y estabilidad ante semillas |
# | 6 | Ejecutar `entregable3.ipynb` | — | Capítulo 5 (comparación de modelos) |
# | 7 | `python scripts/ensamble_padre.py` | 25 min | Pronósticos de PADRE y del ensamble en 2024 y 2025 |
# | 8 | `python scripts/ensamble_semillas.py`, `padre_anidado.py --optimizador ...`, `padre_optimizadores.py`, `ensamble_optimizadores.py` | 5 h | Sección 6.7 |
# | 9 | `python scripts/padre_complejidad.py` | 20 min | Complejidad empírica (sección 6.1) |
# | 10 | `python scripts/padre_tabla_pruebas.py`, `padre_vientos_fuertes.py`, `padre_robustez.py`, `padre_final.py` | 1 h | Capítulo 8 y artefactos |
# | 11 | `python -m pytest tests -q` | 2 s | Pruebas unitarias del modelo original |
# | 12 | Ejecutar `modelo_propuesto.ipynb`, `validacion.ipynb` y este notebook | 5 min | Capítulos 6, 8 y 10 |

# %% [markdown]
# ## 10.3 Tabla maestra de experimentos
#
# El universo de `140` combinaciones suma `112` de clasificación (con técnicas de balanceo) y `28` de regresión;
# como nuestra tarea es de regresión, las que corresponden son esas `28`.
# La tabla reúne las 28 combinaciones de modelo y optimizador del capítulo 5, las dos búsquedas con eliminación sucesiva
# (Random Forest y XGBoost) y el modelo original, con la métrica del bucle
# externo (media ± desviación estándar de los cuatro trimestres de 2024), el resultado de 2025, el tiempo de
# búsqueda y la semilla. La tabla se guarda en `resultados/e3/tabla_maestra_final.csv`.

# %%
tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
ENS = RESULTADOS / "ensamble_padre"
val = pd.read_csv(ENS / "validacion_resumen.csv", index_col=0)

# MAE del bucle externo del ensamble, con las predicciones guardadas de sus cinco componentes
TR, TE, F = cargar()
OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)
nom = lambda m, o, n: f"{m}__{o}__{n}".replace(" ", "_").replace("(", "").replace(")", "")
mae = []
for n, _, va in pliegues_externos(TR):
    comp = [np.load(RESULTADOS / "corridas" / f"{nom(m, OPT[m], n)}.npy") for m in ["Random Forest", "XGBoost", "SVR"]]
    comp += [np.load(ENS / f"PADRE_{c}__{n}.npy") for c in ["XGB", "RF"]]
    mae.append(np.mean(np.abs(TR.y.to_numpy()[va] - np.mean(comp, 0))))
p25 = np.mean([np.load(RESULTADOS / "corridas" / f"{nom(m, OPT[m], 'final')}.npy") for m in
               ["Random Forest", "XGBoost", "SVR"]] + [np.load(ENS / f"PADRE_{c}__final.npy") for c in ["XGB", "RF"]], 0)
y25 = TE.y.to_numpy()
original = {"modelo": "Ensamble PADRE (5) — modelo original", "optimizador": "el de cada componente",
            "RMSE externo": f"{val.loc['Ensamble PADRE (5)', 'RMSE bucle externo (media)']:.4f} ± "
                            f"{val.loc['Ensamble PADRE (5)', 'RMSE bucle externo (DE)']:.4f}",
            "MAE externo": float(np.mean(mae)), "RMSE 2025": float(np.sqrt(np.mean((y25 - p25) ** 2))),
            "MAE 2025": float(np.mean(np.abs(y25 - p25))),
            "R² 2025": float(1 - np.sum((y25 - p25) ** 2) / np.sum((y25 - y25.mean()) ** 2)),
            "evaluaciones": "—", "minutos de búsqueda": "sin búsqueda propia (4 pares y 20 anclas fijados en 2024)",
            "elegido": True}
maestra = pd.concat([tm.drop(columns=["configuración final", "RMSE interno (final)"]), pd.DataFrame([original])],
                    ignore_index=True).assign(semilla=SEED)
maestra.to_csv(RESULTADOS / "tabla_maestra_final.csv", index=False)
maestra.style.format({"MAE externo": "{:.4f}", "RMSE 2025": "{:.4f}", "MAE 2025": "{:.4f}", "R² 2025": "{:.4f}",
                      "minutos de búsqueda": lambda v: f"{v:.1f}" if isinstance(v, float) else v}).hide(axis="index")

# %% [markdown]
# ## 10.4 Artefactos
#
# `scripts/padre_final.py` reentrena el modelo original con 2020–2024 y serializa con joblib (`compress=3`) en
# `artifacts/` los componentes livianos, cada uno con un `metadata.json` (versiones, semilla, hiperparámetros,
# esquema de columnas con sus rangos y métricas del bucle externo). La prueba de ida y vuelta comprueba que el
# modelo cargado da exactamente las mismas predicciones que el modelo en memoria sobre las primeras `5 000` filas de
# 2025. Random Forest y PADRE-RF no se serializan: con `232` árboles de profundidad `21` ocupan varios GB (PADRE-RF
# se entrena con cuatro veces más filas), más de lo que admite un repositorio; se reconstruyen con el mismo script.

# %%
fin = json.loads((ENS / "final_resumen.json").read_text(encoding="utf-8"))
art = Path("artifacts")
filas = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(art.glob("*.metadata.json"))]
print(f"Reentrenamiento: ajuste {fin['segundos_ajuste'] / 60:.1f} min | RMSE 2025 {fin['rmse_2025']:.4f} | "
      f"diferencia máxima con los pronósticos usados en el libro: {fin['max_dif_cache']:.3f} m/s")
pd.DataFrame([{"artefacto": m["modelo"], "tamaño (MB)": m["tamano_MB"], "prueba de ida y vuelta": m["prueba_ida_y_vuelta"],
               "columnas de entrada": len(m["esquema_entrada"])} for m in filas]).set_index("artefacto")

# %% [markdown]
# Los tres artefactos pasan la prueba de ida y vuelta. El reentrenamiento reproduce exactamente los pronósticos de
# XGBoost y de PADRE-XGB usados en el libro, pero no los del SVR: su pronóstico de 2025 cambia hasta `0.33 m/s` en
# algunas filas (RMSE `0.7304` frente a `0.7311` en el capítulo 5), y por eso el ensamble reentrenado difiere hasta
# `0.08 m/s` del que usamos (RMSE `0.7194` frente a `0.7193`). La causa es numérica. Con `γ = 0.001` el núcleo RBF
# entre los `300` puntos de Nyström es casi constante; su matriz tiene número de condición `≈ 1.7 × 10⁸`, y al
# invertirla pequeñas diferencias de redondeo entre ejecuciones se amplifican.
# Todas las conclusiones se mantienen con cualquiera de las dos versiones. Los resultados del libro usan los
# pronósticos guardados en `resultados/e3/corridas/`.

# %% [markdown]
# ## 10.5 Bibliografía
#
# Las referencias de cada capítulo aparecen al final de su página. Estas son las que usa el libro sobre el modelo
# original, su validación y sus herramientas: {cite}`tynes2021,krogh1995,wolpert1992,liu1999,delle2013,wetzel2022,hajjem2014,rodriguez2006,wooldridge2010,breiman2001,chen2016,pedregosa2011,fortin2012,akiba2019,lundberg2017,diebold1995,harvey1997,giacomini2006,clark2007,hansen2011,hansen2005,white2000,politis1994,holm1979,romano2006,nadeau2003,ideam2025`.
# El archivo BibTeX es `references.bib`.

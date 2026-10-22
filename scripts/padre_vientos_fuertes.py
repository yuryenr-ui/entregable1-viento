"""Respuesta a la observación del Entregable 1: sesgo en vientos fuertes y dependencia de los residuos.

Uso (desde la carpeta entregable1, después de ``ensamble_padre.py``):
    python scripts/padre_vientos_fuertes.py

El docente pidió intentar corregir, con un modelo no lineal, el sesgo del modelo lineal en vientos fuertes y la
dependencia que quedó en sus residuos. Comparamos en 2025 (prueba única) el modelo lineal (Ridge, equivalente al SVR
lineal del Entregable 1: RMSE 0.733 en ambos), los dos mejores no lineales del curso y el modelo original:

1. ``vientos_fuertes_por_observado.csv``: sesgo medio (pronóstico − observado) y RMSE por rango de velocidad
   observada. Condicionar en lo observado siempre muestra subestimación de los extremos en cualquier pronóstico de
   error cuadrático (regresión a la media), así que lo que importa es cuánto se reduce.
2. ``vientos_fuertes_por_pronostico.csv``: calibración, observado medio menos pronóstico medio por rango de
   pronóstico; un pronóstico calibrado tiene sesgo cercano a 0 en todos los rangos.
3. ``residuos_dependencia.csv``: Ljung-Box (24 rezagos) y BDS por estación sobre los residuos horarios, y la
   autocorrelación media de los residuos a 1 y 24 horas.

Resultados en resultados/e3/ensamble_padre/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
warnings.filterwarnings("ignore")
from e3 import evaluacion as ev                                     # noqa: E402
from padre_tabla_pruebas import ENSAMBLE, SALIDA, TE, TR, predicciones  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
MODELOS = ["Ridge", "Random Forest", "XGBoost", ENSAMBLE]
RANGOS = [0, 1, 2, 4, 6, np.inf]


def por_rango(clave, y, P):
    """Sesgo y RMSE de cada modelo por rango de ``clave`` (observado o pronóstico de ese modelo)."""
    filas = []
    for m in MODELOS:
        ref = y if clave == "observado" else P[m]
        r = pd.cut(ref, RANGOS, right=False)
        e = pd.DataFrame({"r": r, "err": P[m] - y})
        for rango, g in e.groupby("r", observed=True):
            filas.append({"modelo": m, f"{clave} (m/s)": str(rango), "n": len(g),
                          "sesgo (pronóstico − observado)": g.err.mean(), "RMSE": np.sqrt(np.mean(g.err ** 2))})
    return pd.DataFrame(filas)


def acf_media(est, t, r, rezago):
    """Autocorrelación de los residuos a ``rezago`` horas, promediada sobre estaciones (horas consecutivas)."""
    df = pd.DataFrame({"est": est, "t": pd.to_datetime(t), "r": r}).sort_values(["est", "t"])
    vals = []
    for _, g in df.groupby("est"):
        s = g.set_index("t").r.asfreq("h")
        if s.notna().sum() > 500:
            vals.append(s.autocorr(rezago))
    return float(np.nanmean(vals))


if __name__ == "__main__":
    y25, t25, _, P = predicciones([("final", np.arange(len(TR)), None)])
    est25 = TE.est.to_numpy()
    q95 = float(np.quantile(TR.y, 0.95))
    A = por_rango("observado", y25, P)
    B = por_rango("pronóstico", y25, P)
    fuertes = y25 >= q95
    res = pd.DataFrame([{"modelo": m, "sesgo en el 5 % más fuerte": float(np.mean(P[m][fuertes] - y25[fuertes])),
                         "RMSE en el 5 % más fuerte": float(np.sqrt(np.mean((P[m][fuertes] - y25[fuertes]) ** 2))),
                         "RMSE global": float(np.sqrt(np.mean((P[m] - y25) ** 2)))} for m in MODELOS]).set_index("modelo")
    dep = []
    for m in MODELOS:
        r = y25 - P[m]
        pi = ev.pruebas_independencia(est25, t25, r)
        dep.append({"modelo": m, "estaciones probadas": len(pi),
                    "% estaciones con Ljung-Box p < 0.05": 100 * float((pi.p_LjungBox < 0.05).mean()),
                    "% estaciones con BDS p < 0.05": 100 * float((pi.p_BDS < 0.05).mean()),
                    "ACF media a 1 h": acf_media(est25, t25, r, 1), "ACF media a 24 h": acf_media(est25, t25, r, 24)})
    D = pd.DataFrame(dep).set_index("modelo")
    A.to_csv(SALIDA / "vientos_fuertes_por_observado.csv", index=False)
    B.to_csv(SALIDA / "vientos_fuertes_por_pronostico.csv", index=False)
    res.assign(umbral_m_s=q95).to_csv(SALIDA / "vientos_fuertes_resumen.csv")
    D.to_csv(SALIDA / "residuos_dependencia.csv")
    print(f"Umbral del 5 % más fuerte (percentil 95 del entrenamiento): {q95:.2f} m/s")
    print(A.round(4).to_string(), "\n", B.round(4).to_string(), "\n", res.round(4).to_string(), "\n", D.round(4).to_string())

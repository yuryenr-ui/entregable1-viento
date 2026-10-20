"""Compara PADRE por estación + XGBoost con los 7 modelos del curso, con las mismas pruebas del Entregable 3.

Uso (desde la carpeta entregable1):
    python scripts/padre_comparacion.py 2024 [carpeta]   # 4 trimestres de 2024 (bucle externo)
    python scripts/padre_comparacion.py 2025 [carpeta]   # evaluación única en 2025 (requiere --final)

``carpeta`` es la salida de ``padre_anidado.py`` (``padre_anidado`` por defecto, ``padre_anidado_a80`` para el
espacio ampliado). Cada modelo del curso entra con el optimizador que eligió su bucle externo en el
Entregable 3. Pruebas sobre la pérdida cuadrática diaria: Diebold-Mariano con la corrección de Harvey,
Leybourne y Newbold y ajuste de Holm, Giacomini-White e intervalo de bootstrap estacionario de la diferencia,
y el Model Confidence Set con α = 0.10.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import evaluacion as ev                                       # noqa: E402
from e3.datos import RESULTADOS, cargar, pliegues_externos           # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
PROPUESTO = "PADRE por estación + XGBoost"


def elegidos():
    """Optimizador elegido por el bucle externo para cada modelo (columna ``elegido`` de la tabla resumen)."""
    t = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
    return dict(t[t.elegido].set_index("modelo").optimizador)


def archivo(modelo, optimizador, pliegue):
    nombre = f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")
    return RESULTADOS / "corridas" / f"{nombre}.npy"


if __name__ == "__main__":
    periodo = sys.argv[1]
    carpeta = RESULTADOS / (sys.argv[2] if len(sys.argv) > 2 else "padre_anidado")
    TR, TE, F = cargar()
    opt = elegidos()
    if periodo == "2024":
        partes = [(n, TR.iloc[va]) for n, _, va in pliegues_externos(TR)]
    else:
        partes = [("final", TE)]
    t = np.concatenate([d.t.to_numpy() for _, d in partes])
    y = np.concatenate([d.y.to_numpy() for _, d in partes])
    preds = {PROPUESTO: np.concatenate([np.load(carpeta / f"PADRE_XGB__{n}.npz")["mezcla"] for n, _ in partes]),
             "PADRE por estación (solo)": np.concatenate([np.load(carpeta / f"PADRE_XGB__{n}.npz")["padre"]
                                                          for n, _ in partes])}
    for m, o in opt.items():
        preds[m] = np.concatenate([np.load(archivo(m, o, n)) for n, _ in partes]).astype(float)

    met = pd.DataFrame({k: ev.metricas(y, p) for k, p in preds.items()}).T.sort_values("RMSE")
    met["reducción RMSE vs cada uno (%)"] = 100 * (1 - met.loc[PROPUESTO, "RMSE"] / met.RMSE)
    print(f"Periodo {periodo}: {len(y)} filas\n", met.round(4).to_string(), "\n")

    L = ev.perdida_diaria(t, y, preds)
    filas = []
    for m in opt:
        d = L[PROPUESTO] - L[m]
        dm, gw = ev.diebold_mariano(L[PROPUESTO], L[m]), ev.giacomini_white(L[PROPUESTO], L[m])
        lo, hi = ev.ic_bootstrap_estacionario(d)
        filas.append({"rival": m, "dif. media pérdida": d.mean(), "IC 95 %": f"{lo:.4f} – {hi:.4f}",
                      "DM-HLN": dm["DM_HLN"], "p DM": dm["p_valor"], "GW": gw["GW"], "p GW": gw["p_valor"]})
    cmp = pd.DataFrame(filas).set_index("rival")
    cmp["p DM (Holm)"] = multipletests(cmp["p DM"], method="holm")[1]
    cmp["p GW (Holm)"] = multipletests(cmp["p GW"], method="holm")[1]
    print(f"{len(L)} días. Pérdida diaria propuesto − rival (negativo favorece al propuesto)\n",
          cmp.round(4).to_string(), "\n")
    mcs = ev.mcs(L.drop(columns="PADRE por estación (solo)"))
    print("Model Confidence Set (α = 0.10)\n", mcs.round(4).to_string())
    sal = RESULTADOS / f"padre_comparacion_{periodo}_{carpeta.name}"
    met.to_csv(f"{sal}_metricas.csv")
    cmp.to_csv(f"{sal}_pruebas.csv")
    mcs.to_csv(f"{sal}_mcs.csv")

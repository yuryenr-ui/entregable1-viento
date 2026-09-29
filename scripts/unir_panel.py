"""Une los paneles horarios de cada variable con el catálogo de estaciones del IDEAM.

Uso: python unir_panel.py <catalogo.csv> [--causal]
Entrada: panel_velocidad.csv (obligatorio) y panel_direccion/temperatura/presion.csv (los que existan);
con --causal, los panel_<variable>_causal.csv.
Salida: panel_multivariado.csv (una fila por estación-hora con velocidad válida), o
panel_multivariado_causal.csv con --causal. En el modo causal no se aplica el control anual de presión
contra la altitud: la limpieza causal ya lo hace lectura por lectura con información previa.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAX_DIF_ALTITUD = 50  # hPa: presión mediana de una estación-año incompatible con la altitud de la estación
CAUSAL = "--causal" in sys.argv
SUF = "_causal" if CAUSAL else ""

cat = pd.read_csv(sys.argv[1], dtype=str)
cat["altitud"] = pd.to_numeric(cat.Altitud.str.replace(".", "", regex=False).str.replace(",", "."), errors="coerce")
cat = cat.rename(columns={"Codigo": "CodigoEstacion", "Categoria": "categoria"})[
    ["CodigoEstacion", "categoria", "altitud"]]


def cargar(var):
    p = pd.read_csv(f"panel_{var}{SUF}.csv", dtype={"CodigoEstacion": str}, parse_dates=["hora"])
    return p


base = cargar("velocidad").rename(columns={"v": "vel"}).drop(columns=["CodigoSensor", "n"])

extras = {"direccion": {"v": "dir", "v_sin": "dir_sin", "v_cos": "dir_cos", "constancia": "dir_constancia"},
          "temperatura": {"v": "temp"}, "presion": {"v": "pres"}}
for var, nombres in extras.items():
    if not Path(f"panel_{var}{SUF}.csv").exists():
        print(f"{var}: no hay panel todavía")
        continue
    p = cargar(var).rename(columns=nombres)[["CodigoEstacion", "hora"] + list(nombres.values())]
    base = base.merge(p, on=["CodigoEstacion", "hora"], how="left")

base = base.merge(cat, on="CodigoEstacion", how="left")
print(f"Estaciones sin registro en el catálogo: {base[base.altitud.isna()].CodigoEstacion.nunique()}")

# control cruzado: presión en estación vs. presión esperada por la altitud (atmósfera estándar)
if "pres" in base and not CAUSAL:
    base["anio"] = base.hora.dt.year
    esperada = 1013.25 * (1 - 2.25577e-5 * base.altitud) ** 5.25588
    med = base.groupby(["CodigoEstacion", "anio"]).pres.transform("median")
    dif = med - esperada
    malas = dif.abs() > MAX_DIF_ALTITUD
    tabla = base[malas].groupby(["CodigoEstacion", "nombre", "anio"]).agg(
        altitud=("altitud", "first"), pres_mediana=("pres", "median")).reset_index()
    print("Estación-año con presión incompatible con su altitud (presión descartada):")
    print(tabla.to_string(index=False) if len(tabla) else "  ninguna")
    base.loc[malas, "pres"] = np.nan
    print(f"Diferencia mediana |presión − esperada por altitud|: {dif[~malas].abs().median():.1f} hPa")
    base = base.drop(columns="anio")

base = base.sort_values(["CodigoEstacion", "hora"])
base.to_csv(f"panel_multivariado{SUF}.csv", index=False)

base["anio"] = base.hora.dt.year
cols = [c for c in ["vel", "dir", "temp", "pres"] if c in base]
disp = base[cols].notna()
res = pd.DataFrame({c: disp[c].groupby(base.anio).mean() for c in cols}).round(3)
res["con las 4"] = disp.all(axis=1).groupby(base.anio).sum()
res["estaciones con las 4"] = base[disp.all(axis=1)].groupby("anio").CodigoEstacion.nunique()
print(f"\nPanel multivariado: {len(base):,} estación-horas con velocidad | {base.CodigoEstacion.nunique()} estaciones")
print("Fracción de horas con cada variable disponible, por año:")
print(res.to_string())

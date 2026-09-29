"""Revisa en los CSV crudos de velocidad las lecturas altas que el umbral de control elimina.

Uso: python revisar_umbral.py "<carpeta VELOCIDAD DEL VIENTO>"
Salida: resultados/umbral_velocidad.csv (un resumen pequeño que lee el notebook).

Para cada lectura por encima de 15 m/s se guarda la lectura anterior y la siguiente del mismo sensor y
el tiempo que las separa. Una lectura es un "pico aislado" si (i) las dos vecinas están a no más de
2 pasos de muestreo del sensor (20 min con sensores de 10 min, 4 min con los de 2 min) y (ii) la lectura
supera en más de 15 m/s a la mediana de sus dos vecinas. Es un indicio de anomalía del sensor, no una
prueba definitiva.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEPTOS = {"ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA", "MAGDALENA", "SUCRE"}
FALLAS = {"0014015010", "0015065190"}  # Galerazamba y Mongui: sensores excluidos en la auditoría
UMBRALES = [20, 25, 30]

altas = []
for f in sorted(Path(sys.argv[1]).glob("*.csv")):
    d = pd.read_csv(f, dtype=str, usecols=["CodigoEstacion", "CodigoSensor", "FechaObservacion", "ValorObservado",
                                           "Departamento"])
    d = d[d.Departamento.isin(DEPTOS)]
    d["v"] = pd.to_numeric(d.ValorObservado.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                           errors="coerce")
    d["t"] = pd.to_datetime(d.FechaObservacion, format="%Y %b %d %I:%M:%S %p", errors="coerce")
    d = d.dropna(subset=["v", "t"]).drop_duplicates(["CodigoEstacion", "CodigoSensor", "t"])
    d = d.sort_values(["CodigoEstacion", "CodigoSensor", "t"])
    g = d.groupby(["CodigoEstacion", "CodigoSensor"])
    d["v_ant"], d["v_sig"] = g.v.shift(1), g.v.shift(-1)
    d["dt_ant"] = (d.t - g.t.shift(1)).dt.total_seconds() / 60
    d["dt_sig"] = (g.t.shift(-1) - d.t).dt.total_seconds() / 60
    d["paso"] = d.groupby(["CodigoEstacion", "CodigoSensor"]).dt_ant.transform("median")
    altas.append(d[d.v > 15][["CodigoEstacion", "t", "v", "v_ant", "v_sig", "dt_ant", "dt_sig", "paso"]])
    print(f"{f.name}: {len(d):,} lecturas | > 15 m/s: {(d.v > 15).sum():,}", flush=True)

a = pd.concat(altas, ignore_index=True)
a["sensor con falla"] = np.where(a.CodigoEstacion.isin(FALLAS), "Galerazamba o Mongui", "otras estaciones")
a["vecinas cercanas"] = (a.dt_ant <= 2 * a.paso) & (a.dt_sig <= 2 * a.paso)
a["pico aislado"] = a["vecinas cercanas"] & ((a.v - a[["v_ant", "v_sig"]].median(axis=1)) > 15)

filas = []
for u in UMBRALES:
    for grupo, g in a[a.v > u].groupby("sensor con falla"):
        filas.append({"umbral (m/s)": u, "estaciones": grupo, "lecturas eliminadas": len(g),
                      "estaciones distintas": g.CodigoEstacion.nunique(),
                      "% con vecinas a ≤ 2 pasos": round(100 * g["vecinas cercanas"].mean(), 1),
                      "% picos aislados": round(100 * g["pico aislado"].mean(), 1),
                      "mediana de las lecturas vecinas (m/s)": round(g[["v_ant", "v_sig"]].median(axis=1).median(), 1)})
res = pd.DataFrame(filas)
Path("resultados").mkdir(exist_ok=True)
res.to_csv("resultados/umbral_velocidad.csv", index=False)
print(res.to_string(index=False))

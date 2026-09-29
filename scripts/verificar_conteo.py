"""Verificación independiente del conteo estación-hora, directo desde los CSV crudos y sin limpieza.

Para cada variable y año: conjunto de (estación, hora) con al menos una lectura. Luego se cruzan:
  a) por HORA (lo que hace el pipeline: cada lectura se asigna a su hora)
  b) por INSTANTE EXACTO (exigir la misma marca de tiempo al minuto en las 4 variables)
"""
import sys
from pathlib import Path

import pandas as pd

raiz = Path(sys.argv[1])
carpetas = {"vel": "VELOCIDAD DEL VIENTO", "dir": "DIRECCION DEL VIENTO", "temp": "TEMPERATURA", "pres": "PRESION"}
filas = []
for anio in range(2020, 2026):
    horas, instantes = {}, {}
    for var, sub in carpetas.items():
        f = next((raiz / sub).glob(f"*{anio}.csv"))
        d = pd.read_csv(f, dtype=str, usecols=["CodigoEstacion", "FechaObservacion", "Departamento"])
        d = d[d.Departamento != "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA"]
        t = pd.to_datetime(d.FechaObservacion, format="%Y %b %d %I:%M:%S %p")
        instantes[var] = set(zip(d.CodigoEstacion, t))
        horas[var] = set(zip(d.CodigoEstacion, t.dt.floor("h")))
    por_hora = set.intersection(*horas.values())
    por_instante = set.intersection(*instantes.values())
    filas.append({"año": anio, **{f"horas {v}": len(s) for v, s in horas.items()},
                  "4 variables, misma HORA": len(por_hora), "4 variables, mismo INSTANTE": len(por_instante),
                  "estaciones (misma hora)": len({e for e, _ in por_hora})})
    print(filas[-1], flush=True)

t = pd.DataFrame(filas).set_index("año")
t.loc["total"] = t.sum()
print(t.to_string())

"""Limpieza CAUSAL y panel horario de una variable IDEAM: cada lectura se limpia usando solo información
anterior a ella (o metadatos estáticos del catálogo). Sirve para evaluar el modelo sin información futura
en ninguna partición.

Uso: python procesar_variable_causal.py <variable> <catalogo.csv> archivo1.csv archivo2.csv ... [--temp-solo-altas]
     <variable> ∈ velocidad | direccion | temperatura | presion
Salida: panel_<variable>_causal.csv y reportes/<variable>_causal.txt

Reglas (mismos parámetros por variable que procesar_variable.py):
* Rango de control fijo, lectura por lectura. Todos los umbrales se fijaron antes de ver los datos de 2025.
* Sensor defectuoso: la lectura se enmascara si, en los 30 días ANTERIORES al día de la lectura, más del
  0.5 % de las lecturas del sensor estuvieron fuera de rango. Se
  exigen al menos 100 lecturas en esa ventana.
* Sensor pegado: el valor se enmascara solo desde el momento en que el tramo constante alcanza el plazo de
  su variable (6 h; 12 h en presión; 24 h si el valor es 0 en velocidad y dirección). Las lecturas previas
  del tramo se conservan: en tiempo real ya se habrían usado.
* Presión: referencia = mediana de las medianas diarias de los 30 días anteriores (mínimo 3 días); sin
  historial, la presión estándar para la altitud del catálogo. Picos a > 30 hPa de la referencia → NaN. Si
  la referencia histórica se aleja > 50 hPa de la esperada por la altitud, la lectura se descarta.
* Frecuencia de muestreo: mediana de los 144 intervalos anteriores del sensor (mínimo 12); sin historial,
  la frecuencia nominal del tipo de sensor.
* Varios sensores en la misma estación y hora: se promedian (no se elige ninguno con información futura).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "velocidad": dict(rango=(0, 25), pegado_h=6, cero_h=24, circular=False, solo_altas=False),
    "direccion": dict(rango=(0, 360), pegado_h=6, cero_h=24, circular=True, solo_altas=False),
    # criterio de sensor defectuoso: toda lectura fuera de [3, 45] °C (fijado antes de ver 2025)
    "temperatura": dict(rango=(3, 45), pegado_h=6, cero_h=None, circular=False, solo_altas=False),
    "presion": dict(rango=(500, 1100), pegado_h=12, cero_h=None, circular=False, solo_altas=False, desvio=30),
}
NOMINAL_MIN = {"0103": 10, "0111": 2, "0104": 10, "0068": 60, "0071": 2, "0255": 60, "0258": 2}
DEPTOS = {"ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA", "MAGDALENA", "SUCRE"}
FRAC_DEFECTO, VENTANA, MIN_LECT_VENTANA = 0.005, "30D", 100
MAX_DIF_ALTITUD, MIN_FRAC_HORA = 50, 0.5

VAR, CATALOGO = sys.argv[1], sys.argv[2]
ARCHIVOS = [x for x in sys.argv[3:] if not x.startswith("--")]
CFG = dict(CONFIG[VAR])
# análisis de sensibilidad: en temperatura, contar solo las lecturas > 45 °C para marcar un sensor como
# defectuoso (criterio de la limpieza original, adoptado después de ver 2025)
if "--temp-solo-altas" in sys.argv:
    CFG["solo_altas"] = True
SUF = "_causal_alt" if "--temp-solo-altas" in sys.argv else "_causal"
LO, HI = CFG["rango"]
Path("reportes").mkdir(exist_ok=True)
log_lineas = []


def log(s):
    print(s, flush=True)
    log_lineas.append(s)


def numero(s):
    return pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")


# ---------- lectura de todos los años (la historia debe cruzar el cambio de año) ----------
partes = []
for f in ARCHIVOS:
    d = pd.read_csv(f, dtype=str, usecols=["CodigoEstacion", "CodigoSensor", "FechaObservacion", "ValorObservado",
                                           "NombreEstacion", "Departamento", "Municipio", "Latitud", "Longitud"])
    d = d[d.Departamento.isin(DEPTOS)]
    d["v"] = numero(d.ValorObservado).astype("float32")
    d["t"] = pd.to_datetime(d.FechaObservacion, format="%Y %b %d %I:%M:%S %p", errors="coerce")
    partes.append(d.drop(columns=["ValorObservado", "FechaObservacion"]))
    log(f"{Path(f).name}: {len(d):,} lecturas del Caribe continental")
d = pd.concat(partes, ignore_index=True)
del partes
clave = ["CodigoEstacion", "CodigoSensor", "t"]
n0 = len(d)
d = d.dropna(subset=["t", "v"]).drop_duplicates(clave).sort_values(clave).reset_index(drop=True)
log(f"Lecturas: {n0:,} | tras quitar duplicados y valores no numéricos: {len(d):,}")

meta = d.groupby("CodigoEstacion").agg(nombre=("NombreEstacion", "first"), depto=("Departamento", "first"),
                                       municipio=("Municipio", "first"), Latitud=("Latitud", "first"),
                                       Longitud=("Longitud", "first"))  # primera observación: sin información futura
meta["lat"] = pd.to_numeric(meta.Latitud.str.replace(",", "."), errors="coerce")
meta["lon"] = pd.to_numeric(meta.Longitud.str.replace(",", "."), errors="coerce")
d = d.drop(columns=["NombreEstacion", "Departamento", "Municipio", "Latitud", "Longitud"])
for c in ["CodigoEstacion", "CodigoSensor"]:
    d[c] = d[c].astype("category")

g = d.groupby(["CodigoEstacion", "CodigoSensor"], observed=True, sort=False)
d["dt"] = g.t.diff().dt.total_seconds().div(60).astype("float32")

# ---------- frecuencia de muestreo con historia previa ----------
paso_hist = g.dt.transform(lambda s: s.rolling(144, min_periods=12).median())
nominal = d.CodigoSensor.astype(str).map(NOMINAL_MIN).astype("float32")
d["paso"] = paso_hist.fillna(nominal).fillna(10).clip(lower=1).astype("float32")

# ---------- criterio de lectura anómala (umbral fijo) ----------
fuera = ~d.v.between(LO, HI)
criterio = (d.v > HI) if CFG["solo_altas"] else fuera

# ---------- sensor defectuoso según los 30 días anteriores ----------
d["dia"] = d.t.dt.floor("D")
dia = (d.assign(c=criterio).groupby(["CodigoEstacion", "CodigoSensor", "dia"], observed=True)
       .agg(n=("v", "size"), malas=("c", "sum")).reset_index().sort_values(["CodigoEstacion", "CodigoSensor", "dia"]))
roll = (dia.set_index("dia").groupby(["CodigoEstacion", "CodigoSensor"], observed=True)[["n", "malas"]]
        .rolling(VENTANA, closed="left").sum().reset_index())
roll["defectuoso"] = (roll.n >= MIN_LECT_VENTANA) & (roll.malas / roll.n > FRAC_DEFECTO)
d = d.merge(roll[["CodigoEstacion", "CodigoSensor", "dia", "defectuoso"]], on=["CodigoEstacion", "CodigoSensor", "dia"],
            how="left")
d["defectuoso"] = d.defectuoso.fillna(False).astype(bool)

# ---------- tramos pegados detectados en tiempo real ----------
g = d.groupby(["CodigoEstacion", "CodigoSensor"], observed=True, sort=False)
nuevo = (d.v != g.v.shift()) | (d.dt > 2 * d.paso) | d.dt.isna()
tramo = nuevo.cumsum()
transcurrido_h = (d.t - d.t.groupby(tramo).transform("first")).dt.total_seconds() / 3600
if CFG["cero_h"] is None:
    pegado = transcurrido_h >= CFG["pegado_h"]
else:
    pegado = ((d.v != 0) & (transcurrido_h >= CFG["pegado_h"])) | ((d.v == 0) & (transcurrido_h >= CFG["cero_h"]))

# ---------- presión: referencia histórica y control con la altitud ----------
picos = pd.Series(False, index=d.index)
altitud_mala = pd.Series(False, index=d.index)
if "desvio" in CFG:
    cat = pd.read_csv(CATALOGO, dtype=str)
    alt = pd.to_numeric(cat.set_index("Codigo").Altitud.str.replace(".", "", regex=False).str.replace(",", "."),
                        errors="coerce")
    esperada = 1013.25 * (1 - 2.25577e-5 * alt) ** 5.25588
    med_dia = (d[~fuera].groupby(["CodigoEstacion", "CodigoSensor", "dia"], observed=True).v.median().reset_index()
               .sort_values(["CodigoEstacion", "CodigoSensor", "dia"]))
    ref = (med_dia.set_index("dia").groupby(["CodigoEstacion", "CodigoSensor"], observed=True).v
           .rolling(VENTANA, closed="left", min_periods=3).median().rename("ref").reset_index())
    d = d.merge(ref, on=["CodigoEstacion", "CodigoSensor", "dia"], how="left")
    esp = d.CodigoEstacion.astype(str).map(esperada)
    altitud_mala = (d.ref - esp).abs() > MAX_DIF_ALTITUD
    referencia = d.ref.fillna(esp)
    picos = (d.v - referencia).abs() > CFG["desvio"]
    log(f"Lecturas sin historial de presión (referencia = altitud del catálogo): {d.ref.isna().sum():,}")

d["v_ok"] = d.v.where(~fuera & ~pegado & ~d.defectuoso & ~picos & ~altitud_mala)
n = len(d)
for nombre_, m in [("fuera de rango", fuera), ("sensor defectuoso (30 días previos)", d.defectuoso),
                   ("tramo pegado (desde que cumple el plazo)", pegado), ("pico de presión", picos),
                   ("presión incompatible con la altitud", altitud_mala)]:
    log(f"  {nombre_}: {int(m.sum()):,} lecturas ({100 * m.mean():.3f} %)")
log(f"  enmascaradas en total: {int(d.v_ok.isna().sum()):,} ({100 * d.v_ok.isna().mean():.3f} %)")

# ---------- agregación horaria por sensor y promedio entre sensores ----------
d["hora"] = d.t.dt.floor("h")
llaves = ["CodigoEstacion", "CodigoSensor", "hora"]
if CFG["circular"]:
    rad = np.deg2rad(d.v_ok)
    d["s"], d["c"] = np.sin(rad), np.cos(rad)
    h = d.groupby(llaves, observed=True).agg(s=("s", "mean"), c=("c", "mean"), n=("v_ok", "count"),
                                             paso=("paso", "last")).reset_index()
else:
    h = d.groupby(llaves, observed=True).agg(v=("v_ok", "mean"), n=("v_ok", "count"), paso=("paso", "last")).reset_index()
h = h[h.n >= np.maximum(1, MIN_FRAC_HORA * 60 / h.paso)]
varios = h.groupby(["CodigoEstacion", "hora"], observed=True).size()
log(f"Estación-horas con más de un sensor válido (se promedian): {int((varios > 1).sum()):,}")
if CFG["circular"]:
    h = h.groupby(["CodigoEstacion", "hora"], observed=True).agg(v_sin=("s", "mean"), v_cos=("c", "mean"),
                                                                 n=("n", "sum")).reset_index()
    h["v"] = np.rad2deg(np.arctan2(h.v_sin, h.v_cos)) % 360
    h["constancia"] = np.hypot(h.v_sin, h.v_cos)
    extra = ["v_sin", "v_cos", "constancia"]
else:
    h = h.groupby(["CodigoEstacion", "hora"], observed=True).agg(v=("v", "mean"), n=("n", "sum")).reset_index()
    extra = []
h["CodigoEstacion"] = h.CodigoEstacion.astype(str)
h = h.merge(meta[["nombre", "depto", "municipio", "lat", "lon"]], left_on="CodigoEstacion", right_index=True)
h["CodigoSensor"] = "causal"
h = h.sort_values(["CodigoEstacion", "hora"])
h[["CodigoEstacion", "CodigoSensor", "nombre", "depto", "municipio", "lat", "lon", "hora", "v"] + extra + ["n"]].to_csv(
    f"panel_{VAR}{SUF}.csv", index=False)
log(f"Panel causal {VAR}: {len(h):,} estación-horas | {h.CodigoEstacion.nunique()} estaciones")
log("Estación-horas por año: " + str(h.groupby(h.hora.dt.year).size().to_dict()))
Path("reportes", f"{VAR}{SUF}.txt").write_text("\n".join(log_lineas), encoding="utf-8")

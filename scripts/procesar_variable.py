"""Auditoría, limpieza y panel horario de variables IDEAM (datos.gov.co).

Uso: python procesar_variable.py <variable> archivo1.csv archivo2.csv ...
     <variable> ∈ velocidad | direccion | temperatura | presion

Cada archivo se procesa por separado (para no cargar varios GB a la vez); el año se toma de las
fechas del archivo, no de su nombre. Salidas: panel_<variable>.csv, auditoria_<variable>.csv,
cobertura_<variable>.csv y un reporte por archivo en reportes/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# rango físico válido, regla de valores pegados y tipo de promedio horario por variable
CONFIG = {
    # media de 10 min > 25 m/s solo con ciclón; 0 sostenido 24 h = sensor caído
    "velocidad": dict(rango=(0, 25), pegado_h=6, cero_h=24, circular=False, unidad="m/s"),
    # la dirección se promedia como vector unitario (359° y 1° promedian 0°, no 180°)
    "direccion": dict(rango=(0, 360), pegado_h=6, cero_h=24, circular=True, unidad="grados"),
    # rango amplio (estaciones de la Sierra Nevada y el Perijá > 2 000 m). Los 0 °C sueltos son cortes del
    # sensor y solo se enmascaran; lo que delata un sensor defectuoso es que marque > 45 °C con frecuencia
    "temperatura": dict(rango=(3, 45), pegado_h=6, cero_h=None, circular=False, unidad="°C", excluir_solo_altas=True),
    # presión en estación (no reducida al nivel del mar): baja con la altitud. En el trópico varía ±5 hPa,
    # así que una lectura a > 30 hPa de la mediana de su estación-año es un pico del sensor
    "presion": dict(rango=(500, 1100), pegado_h=12, cero_h=None, circular=False, unidad="hPa", desvio=30),
}
FRAC_FUERA = 0.005   # si > 0.5 % de las lecturas de una estación-año están fuera de rango, el sensor falla
MIN_FRAC_HORA = 0.5  # una hora es válida si tiene al menos la mitad de las lecturas esperadas
DEPTOS = {"ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA", "MAGDALENA", "SUCRE"}  # Caribe continental

VAR = sys.argv[1]
CFG = CONFIG[VAR]
LO, HI = CFG["rango"]
Path("reportes").mkdir(exist_ok=True)


def numero(s):
    """'1.003,4' -> 1003.4 ; '3,7' -> 3.7 (punto = miles, coma = decimal)."""
    return pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")


def leer(f):
    d = pd.read_csv(f, dtype=str, usecols=["CodigoEstacion", "CodigoSensor", "FechaObservacion", "ValorObservado",
                                             "NombreEstacion", "Departamento", "Municipio", "Latitud", "Longitud",
                                             "UnidadMedida"])
    d["v"] = numero(d.ValorObservado)
    d["t"] = pd.to_datetime(d.FechaObservacion, format="%Y %b %d %I:%M:%S %p", errors="coerce")
    d["lat"] = pd.to_numeric(d.Latitud.str.replace(",", "."), errors="coerce")
    d["lon"] = pd.to_numeric(d.Longitud.str.replace(",", "."), errors="coerce")
    return d.drop(columns=["ValorObservado", "FechaObservacion", "Latitud", "Longitud"])


def procesar(f, log):
    d = leer(f)
    anios = sorted(d.t.dt.year.dropna().unique().astype(int))
    log(f"Variable: {VAR} | archivo: {Path(f).name}\nAños en el contenido: {anios}")
    log(f"Filas: {len(d):,} | fechas no parseadas: {d.t.isna().sum()} | valores no numéricos: {d.v.isna().sum()}")
    log(f"Unidades: {d.UnidadMedida.value_counts().to_dict()}")
    fuera = ~d.Departamento.isin(DEPTOS)
    log(f"Filas fuera del Caribe continental (descartadas): {fuera.sum():,} {d[fuera].Departamento.unique().tolist()}")
    d = d[~fuera]
    clave = ["CodigoEstacion", "CodigoSensor", "t"]
    n_dup, n_exac = d.duplicated(clave).sum(), d.duplicated(clave + ["v"]).sum()
    n_conf = d[d.duplicated(clave, keep=False)].groupby(clave).v.nunique().gt(1).sum()
    log(f"Duplicados estación+sensor+instante: {n_dup:,} (idénticos: {n_exac:,}; "
        f"instantes con valores distintos: {n_conf:,}, se conserva el primero)")
    d = d.dropna(subset=["t", "v"]).drop_duplicates(clave)
    d["anio"] = d.t.dt.year
    log("Estaciones por departamento y año:\n" +
        d.groupby(["Departamento", "anio"]).CodigoEstacion.nunique().unstack().to_string())
    log("Distribución de valores: " + str(d.v.describe(percentiles=[.001, .01, .5, .99, .999]).round(2).to_dict()))

    d = d.sort_values(clave).reset_index(drop=True)
    g = d.groupby(["CodigoEstacion", "CodigoSensor"])
    dt_min = g.t.diff().dt.total_seconds().div(60)
    paso = dt_min.groupby([d.CodigoEstacion, d.CodigoSensor]).median().clip(lower=1).rename("paso")
    d = d.merge(paso, left_on=["CodigoEstacion", "CodigoSensor"], right_index=True)
    log("Paso de muestreo (min) → número de estación-sensor: " + str(paso.round().value_counts().to_dict()))

    # tramos pegados: mismo valor en lecturas consecutivas; un hueco > 2 pasos corta el tramo
    run = ((d.v != g.v.shift()) | (dt_min > 2 * d.paso)).cumsum()
    r = d.groupby(run).agg(v=("v", "first"), ini=("t", "first"), fin=("t", "last"))
    r["h"] = (r.fin - r.ini).dt.total_seconds() / 3600
    if CFG["cero_h"] is None:
        malos = r[r.h >= CFG["pegado_h"]].index
    else:
        malos = r[((r.v != 0) & (r.h >= CFG["pegado_h"])) | ((r.v == 0) & (r.h >= CFG["cero_h"]))].index
    pegado = run.isin(malos)

    # estación-año con sensor fallando: demasiadas lecturas fuera del rango físico
    fuera_rango = ~d.v.between(LO, HI)
    if "desvio" in CFG:
        med = d.groupby(["CodigoEstacion", "anio"]).v.transform("median")
        picos = (d.v - med).abs() > CFG["desvio"]
        log(f"Picos a > {CFG['desvio']} {CFG['unidad']} de la mediana de su estación-año (enmascarados): {picos.sum():,}")
        pegado = pegado | picos
    criterio = (d.v > HI) if CFG.get("excluir_solo_altas") else fuera_rango
    ea = d.assign(fuera=criterio, peg=pegado).groupby(["CodigoEstacion", "NombreEstacion", "anio"]).agg(
        lecturas=("v", "size"), frac_fuera=("fuera", "mean"), min_v=("v", "min"), max_v=("v", "max"),
        mediana=("v", "median"), frac_pegado=("peg", "mean"))
    ea["excluida"] = ea.frac_fuera > FRAC_FUERA
    texto = f"> {HI}" if CFG.get("excluir_solo_altas") else f"fuera de [{LO}, {HI}]"
    ea["motivo"] = np.where(ea.excluida, f"> {FRAC_FUERA:.1%} de lecturas {texto} {CFG['unidad']}", "")
    log(f"Lecturas fuera de [{LO}, {HI}] {CFG['unidad']}: {fuera_rango.sum():,} | "
        f"lecturas pegadas o con picos (enmascaradas): {pegado.sum():,}")
    log("Estación-año excluidas:\n" + (ea[ea.excluida].round(3).to_string() if ea.excluida.any() else "  ninguna"))
    top_peg = ea[ea.frac_pegado > 0.05].sort_values("frac_pegado", ascending=False)
    log("Estación-año con > 5 % de lecturas pegadas (enmascaradas):\n" +
        (top_peg.round(3).to_string() if len(top_peg) else "  ninguna"))

    d["v_ok"] = d.v.where(~fuera_rango & ~pegado)
    excl = ea[ea.excluida].reset_index()[["CodigoEstacion", "anio"]].assign(_x=1)
    d = d.merge(excl, on=["CodigoEstacion", "anio"], how="left")
    d = d[d._x.isna()].drop(columns="_x")

    # agregación horaria
    d["hora"] = d.t.dt.floor("h")
    llaves = ["CodigoEstacion", "CodigoSensor", "hora"]
    if CFG["circular"]:
        rad = np.deg2rad(d.v_ok)
        d["s"], d["c"] = np.sin(rad), np.cos(rad)
        h = d.groupby(llaves).agg(s=("s", "mean"), c=("c", "mean"), n=("v_ok", "count"),
                                  paso=("paso", "first")).reset_index()
        h["v"] = np.rad2deg(np.arctan2(h.s, h.c)) % 360
        h["constancia"] = np.hypot(h.s, h.c)  # 1 = dirección fija en la hora, 0 = dirección variable
        h = h.rename(columns={"s": "v_sin", "c": "v_cos"})
    else:
        h = d.groupby(llaves).agg(v=("v_ok", "mean"), n=("v_ok", "count"), paso=("paso", "first")).reset_index()
    h = h[h.n >= np.maximum(1, MIN_FRAC_HORA * 60 / h.paso)]
    meta = d.sort_values("t").groupby("CodigoEstacion").agg(
        nombre=("NombreEstacion", "last"), depto=("Departamento", "last"), municipio=("Municipio", "last"),
        lat=("lat", "median"), lon=("lon", "median"), n_coord=("lat", "nunique"), n_nombres=("NombreEstacion", "nunique"))
    raros = meta[(meta.n_coord > 1) | (meta.n_nombres > 1)]
    log("Códigos con más de un nombre o coordenada:\n" + (raros.to_string() if len(raros) else "  ninguno"))
    h = h.merge(meta[["nombre", "depto", "municipio", "lat", "lon"]], left_on="CodigoEstacion", right_index=True)
    return h.drop(columns="paso"), ea.reset_index()


paneles, auditorias = [], []
for f in sys.argv[2:]:
    lineas = []
    log = lambda s: (print(s), lineas.append(s))
    h, ea = procesar(f, log)
    anios = sorted(h.hora.dt.year.unique())
    Path("reportes", f"{VAR}_{'_'.join(map(str, anios))}.txt").write_text("\n\n".join(lineas), encoding="utf-8")
    paneles.append(h)
    auditorias.append(ea)
    print("-" * 80)

p = pd.concat(paneles, ignore_index=True)
# si dos archivos traen la misma estación-hora, se conserva la de más lecturas
p = p.sort_values("n", ascending=False).drop_duplicates(["CodigoEstacion", "CodigoSensor", "hora"])
# un sensor por estación y año: el de más horas válidas
p["anio"] = p.hora.dt.year
cnt = p.groupby(["CodigoEstacion", "anio", "CodigoSensor"]).size().rename("horas").reset_index()
eleg = cnt.sort_values("horas", ascending=False).drop_duplicates(["CodigoEstacion", "anio"])
print(f"Estación-año con más de un sensor: {(cnt.groupby(['CodigoEstacion', 'anio']).size() > 1).sum()}")
p = p.merge(eleg[["CodigoEstacion", "anio", "CodigoSensor"]], on=["CodigoEstacion", "anio", "CodigoSensor"])
p = p.sort_values(["CodigoEstacion", "hora"])
extra = ["v_sin", "v_cos", "constancia"] if CFG["circular"] else []
p[["CodigoEstacion", "CodigoSensor", "nombre", "depto", "municipio", "lat", "lon", "hora", "v"] + extra + ["n"]].to_csv(
    f"panel_{VAR}.csv", index=False)
pd.concat(auditorias).to_csv(f"auditoria_{VAR}.csv", index=False)

cob = p.groupby(["CodigoEstacion", "nombre", "depto", "anio"]).size().unstack(fill_value=0)
cob.to_csv(f"cobertura_{VAR}.csv")
print(f"\nPanel {VAR}: {len(p):,} estación-horas | {p.CodigoEstacion.nunique()} estaciones")
print("Estación-horas por año:", p.groupby("anio").size().to_dict())
print("Estaciones por año:", p.groupby("anio").CodigoEstacion.nunique().to_dict())

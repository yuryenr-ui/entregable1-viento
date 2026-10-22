"""Tablas obligatorias de la validación estadística del modelo original (resumen y pruebas de hipótesis).

Uso (desde la carpeta entregable1, después de ``ensamble_padre.py``):
    python scripts/padre_tabla_pruebas.py

No reentrena nada: lee las predicciones guardadas de los 7 modelos y de los dos PADRE. Produce, para el bucle
externo (los 4 trimestres de 2024) y para la prueba única (2025):

1. ``validacion_resumen.csv``: por modelo, RMSE medio ± desviación estándar de los 4 trimestres del bucle externo,
   RMSE de 2025 con su IC 95 % por bootstrap estacionario (bloques de 7 días) y tiempos de ajuste y predicción de
   la corrida final.
2. ``validacion_pruebas.csv``: una fila por prueba y rival con las columnas que exige el enunciado (prueba, hipótesis
   nula, supuestos verificados, estadístico, p crudo, p ajustado, tamaño del efecto y conclusión).
3. ``validacion_supuestos.csv``: estacionariedad (ADF y KPSS) y autocorrelación (Ljung-Box) de la diferencia diaria
   de pérdidas frente a cada rival, que es lo que suponen DM, GW, CW, MCS y SPA.

Resultados en resultados/e3/ensamble_padre/.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
warnings.filterwarnings("ignore")
from e3 import evaluacion as ev                                                          # noqa: E402
from ensamble_padre import (ENSAMBLE, OPT, RESULTADOS, SALIDA, TE, TR, F, X, est, nombre,  # noqa: E402
                            padre, pliegues_externos, y)

sys.stdout.reconfigure(encoding="utf-8")
COMPONENTES = ["Random Forest", "XGBoost", "SVR", "PADRE-XGB", "PADRE-RF"]
ALFA = 0.05


def predicciones(partes):
    """Predicciones por pliegue de los 7 modelos y del ensamble (los PADRE salen de la caché)."""
    ys, ts, ns, preds = [], [], [], {}
    for n, tr, va in partes:
        X_ev, y_ev, est_ev, t_ev = ((TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy(), TE.t.to_numpy())
                                    if va is None else (X[va], y[va], est[va], TR.t.to_numpy()[va]))
        p = {m: np.load(RESULTADOS / "corridas" / f"{nombre(m, o, n)}.npy").astype(float) for m, o in OPT.items()}
        p["PADRE-XGB"], p["PADRE-RF"] = padre("XGB", n, tr, X_ev, est_ev), padre("RF", n, tr, X_ev, est_ev)
        p[ENSAMBLE] = np.mean([p[k] for k in COMPONENTES], 0)
        ys.append(y_ev), ts.append(t_ev), ns.append(np.full(len(y_ev), n))
        for k, v in p.items():
            preds.setdefault(k, []).append(v)
    return np.concatenate(ys), np.concatenate(ts), np.concatenate(ns), {k: np.concatenate(v) for k, v in preds.items()}


def supuestos(d):
    """ADF (H0: raíz unitaria), KPSS (H0: estacionaria) y Ljung-Box a 7 rezagos (H0: sin autocorrelación)."""
    d = np.asarray(d)
    return {"p ADF": float(adfuller(d, autolag="AIC")[1]), "p KPSS": float(kpss(d, regression="c", nlags="auto")[1]),
            "p Ljung-Box (7)": float(acorr_ljungbox(d, lags=[7])["lb_pvalue"].iloc[0])}


def texto_supuestos(s):
    est_ = "estacionaria (ADF p < 0.05" + (", KPSS no rechaza)" if s["p KPSS"] >= 0.05 else "; KPSS rechaza al 5 %)")
    if s["p ADF"] >= 0.05:
        est_ = f"ADF no rechaza raíz unitaria (p = {s['p ADF']:.2f})"
    ac = "autocorrelada: varianza HAC / bootstrap por bloques" if s["p Ljung-Box (7)"] < 0.05 else "sin autocorrelación a 7 días"
    return f"diferencia de pérdidas {est_}; {ac}"


def tabla(periodo, t, y_ev, preds):
    L = ev.perdida_diaria(t, y_ev, preds)
    rivales = list(OPT)
    sup = {m: supuestos(L[ENSAMBLE] - L[m]) for m in rivales}
    filas = []
    for m in rivales:                         # DM-HLN y GW: el ensamble frente a cada uno de los 7
        d = L[ENSAMBLE] - L[m]
        dm, gw = ev.diebold_mariano(L[ENSAMBLE], L[m]), ev.giacomini_white(L[ENSAMBLE], L[m])
        efecto = f"d de Cohen = {ev.d_cohen(L[ENSAMBLE], L[m]):.3f}; δ de Cliff = {ev.delta_cliff(L[ENSAMBLE], L[m]):.3f}"
        base = {"periodo": periodo, "rival": m, "supuestos verificados": texto_supuestos(sup[m]), "tamaño del efecto": efecto,
                "d de Cohen": ev.d_cohen(L[ENSAMBLE], L[m]), "dif. media pérdida": float(d.mean())}
        filas.append({**base, "prueba": "Diebold-Mariano (HLN, h = 2)",
                      "hipótesis nula": "igual pérdida cuadrática esperada", "estadístico": dm["DM_HLN"], "p crudo": dm["p_valor"]})
        filas.append({**base, "prueba": "Giacomini-White (incondicional + rezago)",
                      "hipótesis nula": "igual capacidad predictiva condicional", "estadístico": gw["GW"], "p crudo": gw["p_valor"]})
    for ref in ["Random Forest", "XGBoost"]:  # Clark-West: el ensamble contiene a RF y a XGB (modelos anidados)
        dp = ev.perdida_diaria(t, preds[ref], {"d": preds[ENSAMBLE]}).d
        cw = ev.clark_west(L[ref], L[ENSAMBLE], dp)
        filas.append({"periodo": periodo, "rival": ref, "prueba": "Clark-West (anidados)",
                      "hipótesis nula": f"el ensamble no mejora a {ref}, que está anidado en él",
                      "supuestos verificados": "anidamiento: el ensamble contiene a " + ref + "; " + texto_supuestos(sup[ref]),
                      "estadístico": cw["CW"], "p crudo": cw["p_valor"], "tamaño del efecto": "—"})
    T = pd.DataFrame(filas)
    # Holm dentro de cada familia de pruebas (7 rivales para DM y GW, 2 referencias para CW)
    T["p ajustado (Holm)"] = T.groupby("prueba")["p crudo"].transform(lambda p: multipletests(p, method="holm")[1])
    cols = [ENSAMBLE, *rivales]
    mcs = ev.mcs(L[cols])
    spa = ev.spa(L["Random Forest"], L[[c for c in cols if c != "Random Forest"]])
    spa_e = ev.spa(L[ENSAMBLE], L[rivales])
    lo, hi = ev.ic_bootstrap_estacionario(L[ENSAMBLE] - L["Random Forest"])
    sup_g = "diferencias diarias estacionarias (ADF); bootstrap estacionario por bloques de 7 días"
    extra = [
        {"prueba": "Model Confidence Set (α = 0.10)", "rival": "los 8 modelos",
         "hipótesis nula": "todos los modelos del conjunto tienen igual pérdida esperada",
         "estadístico": np.nan, "p crudo": float(mcs.loc[ENSAMBLE, "p_MCS"]),
         "p ajustado (Holm)": float(mcs.loc[ENSAMBLE, "p_MCS"]),
         "tamaño del efecto": "conjunto final: " + ", ".join(mcs.index[mcs["en el conjunto"]])},
        {"prueba": "SPA de Hansen (referencia: Random Forest)", "rival": "los otros 7 (incluido el ensamble)",
         "hipótesis nula": "ningún modelo supera a Random Forest", "estadístico": np.nan,
         "p crudo": spa["p_SPA (Hansen)"], "p ajustado (Holm)": spa["p_SPA (Hansen)"],
         "tamaño del efecto": f"Reality Check de White: p = {spa['p_Reality Check (White)']:.4f}"},
        {"prueba": "SPA de Hansen (referencia: el ensamble)", "rival": "los 7 modelos del curso",
         "hipótesis nula": "ningún modelo del curso supera al ensamble", "estadístico": np.nan,
         "p crudo": spa_e["p_SPA (Hansen)"], "p ajustado (Holm)": spa_e["p_SPA (Hansen)"], "tamaño del efecto": "—"},
        {"prueba": "IC bootstrap estacionario de la diferencia de pérdida", "rival": "Random Forest",
         "hipótesis nula": "la diferencia media de pérdida diaria es 0", "estadístico": float((L[ENSAMBLE] - L["Random Forest"]).mean()),
         "p crudo": np.nan, "p ajustado (Holm)": np.nan, "tamaño del efecto": f"IC 95 %: {lo:.4f} a {hi:.4f}"},
    ]
    for e in extra:
        e.update({"periodo": periodo, "supuestos verificados": sup_g})
    T = pd.concat([T, pd.DataFrame(extra)], ignore_index=True)

    def conclusion(r):
        if r.prueba.startswith("SPA") and "ensamble)" in r.prueba:
            return "no se rechaza: ningún modelo del curso supera al ensamble" if r["p crudo"] >= ALFA else "se rechaza"
        if r.prueba.startswith("Model Confidence"):
            return "el ensamble es el único del conjunto" if r["tamaño del efecto"].endswith(ENSAMBLE) else "el ensamble está en el conjunto"
        if r.prueba.startswith("IC"):
            return "el IC excluye 0: el ensamble pierde menos" if hi < 0 else "el IC incluye 0"
        if r["p ajustado (Holm)"] < ALFA and (r.get("dif. media pérdida", -1) < 0 or r.prueba.startswith(("Clark", "SPA"))):
            return "se rechaza H0: el ensamble es mejor"
        return "no se rechaza H0"
    T["conclusión"] = T.apply(conclusion, axis=1)
    S = pd.DataFrame(sup).T.assign(periodo=periodo)
    return T, S


if __name__ == "__main__":
    p24 = predicciones([(n, tr, va) for n, tr, va in pliegues_externos(TR)])
    p25 = predicciones([("final", np.arange(len(TR)), None)])

    # 1. resumen: media ± DE del bucle externo, RMSE 2025 con IC y tiempos de la corrida final
    y24, _, n24, P24 = p24
    y25, t25, _, P25 = p25
    ic = ev.ic_rmse(t25, y25, P25)
    seg = re.findall(r"PADRE-(\w+) final: (\d+) s", (RESULTADOS / "ensamble_padre.log").read_text(encoding="utf-8"))
    t_padre = {f"PADRE-{k}": float(v) for k, v in seg}
    filas = []
    for m in [ENSAMBLE, *OPT, "PADRE-XGB", "PADRE-RF"]:
        por_pliegue = [np.sqrt(np.mean((y24[n24 == n] - P24[m][n24 == n]) ** 2)) for n in np.unique(n24)]
        if m in OPT:
            js = json.loads((RESULTADOS / "corridas" / f"{nombre(m, OPT[m], 'final')}.json").read_text(encoding="utf-8"))
            ta, tp = js["segundos_ajuste_final"], js["segundos_prediccion"]
        elif m == ENSAMBLE:
            ta = sum(json.loads((RESULTADOS / "corridas" / f"{nombre(c, OPT[c], 'final')}.json").read_text(
                encoding="utf-8"))["segundos_ajuste_final"] for c in COMPONENTES[:3]) + sum(t_padre.values())
            tp = np.nan
        else:
            ta, tp = t_padre.get(m, np.nan), np.nan
        filas.append({"modelo": m, "RMSE bucle externo (media)": np.mean(por_pliegue),
                      "RMSE bucle externo (DE)": np.std(por_pliegue, ddof=1), "RMSE 2025": ic.loc[m, "RMSE"],
                      "IC 95 % inf": ic.loc[m, "IC 95 % inf"], "IC 95 % sup": ic.loc[m, "IC 95 % sup"],
                      "ajuste final (s)": ta, "predicción 2025 (s)": tp})
    R = pd.DataFrame(filas).set_index("modelo")
    R.to_csv(SALIDA / "validacion_resumen.csv")
    print(R.round(4).to_string())

    # 2 y 3. pruebas de hipótesis y supuestos, en el bucle externo y en 2025
    T24, S24 = tabla("2024 (bucle externo)", p24[1], y24, {k: v for k, v in P24.items() if not k.startswith("PADRE-")})
    T25, S25 = tabla("2025 (prueba)", t25, y25, {k: v for k, v in P25.items() if not k.startswith("PADRE-")})
    T = pd.concat([T24, T25], ignore_index=True)
    cols = ["periodo", "prueba", "rival", "hipótesis nula", "supuestos verificados", "estadístico", "p crudo",
            "p ajustado (Holm)", "tamaño del efecto", "conclusión", "d de Cohen", "dif. media pérdida"]
    T[cols].to_csv(SALIDA / "validacion_pruebas.csv", index=False)
    pd.concat([S24, S25]).to_csv(SALIDA / "validacion_supuestos.csv")
    print(T[["periodo", "prueba", "rival", "estadístico", "p crudo", "p ajustado (Holm)", "conclusión"]].round(4).to_string())
    print("\nComparaciones: DM 7 + GW 7 + CW 2 por periodo, con Holm dentro de cada familia; MCS y SPA son globales.")

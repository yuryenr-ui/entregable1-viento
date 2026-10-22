"""Validación anidada del Random Forest de efectos mixtos (MERF), con el mismo esquema del Entregable 2.

Uso (desde la carpeta entregable1):
    python scripts/merf_anidado.py            # 2024 y, si se cumple la regla, la evaluación única en 2025
    python scripts/merf_anidado.py --prueba   # versión mínima para comprobar que todo corre

Esquema (igual que los 7 modelos del curso): bucle externo con los 4 trimestres de 2024, bucle interno con los 3
trimestres anteriores y las mismas submuestras (40 mil filas de ajuste y 10 mil de validación), búsqueda
aleatoria de 40 configuraciones con la semilla 42. El espacio es el del Random Forest del Entregable 2 más dos
hiperparámetros propios: qué coeficientes varían por estación y con cuántos días recientes se estima el efecto de
cada estación. La configuración elegida se reentrena con todo el entrenamiento del pliegue y se mide una vez.

Regla fijada ANTES de ver los resultados (18/10/2026): con los 4 trimestres de 2024 se compara MERF con el
Random Forest del Entregable 2 mediante Diebold-Mariano (HLN) sobre la pérdida diaria. Solo si MERF tiene menor
pérdida y p < 0.05 se ejecuta la búsqueda final y la evaluación única en 2025. Si no, MERF se reporta como
intento y 2025 no se usa. La decisión queda en resultados/e3/merf_anidado/decision.json.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e3 import SEED                                                                     # noqa: E402
from e3 import evaluacion as ev                                                         # noqa: E402
from e3.datos import RESULTADOS, TEST_INI, TRIMESTRES, cargar, particion_interna, pliegues_externos  # noqa: E402
from e3.merf import EFECTOS, MERF                                                       # noqa: E402
from e3.modelos import ESPACIOS                                                         # noqa: E402
from e3.optimizadores import aleatoria                                                  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
VENTANAS = {"toda la historia": None, "último año": 365}
ESPACIO = {**ESPACIOS["Random Forest"], "efectos": ("cat", list(EFECTOS)), "ventana": ("cat", list(VENTANAS))}
ITERACIONES = 3
RF = "Random_Forest__Bayesiana_Optuna"      # el Random Forest elegido por el bucle externo del Entregable 2
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def modelo(params):
    rf = {k: params[k] for k in ESPACIOS["Random Forest"]}
    return MERF(rf, F, params["efectos"], ITERACIONES, VENTANAS[params["ventana"]], n_jobs=16)


def pliegue_merf(pliegue, cfg, carpeta):
    salida = carpeta / f"MERF__{pliegue}.json"
    if salida.exists():
        return json.loads(salida.read_text(encoding="utf-8"))
    if pliegue == "final":
        fin, n_int = TEST_INI, 4
        tr_ext = np.arange(len(TR))
        X_ev, y_ev, est_ev = TE[F].to_numpy(np.float32), TE.y.to_numpy(), TE.est.to_numpy()
    else:
        _, tr_ext, va = next(p for p in pliegues_externos(TR) if p[0] == pliegue)
        fin, n_int = TRIMESTRES[4 + int(pliegue[-1]) - 1], 3
        X_ev, y_ev, est_ev = X[va], y[va], est[va]
    if cfg["prueba"]:
        n_int = 1
    internos = particion_interna(TR, fin, n_int, cfg["n_ajuste"], cfg["n_valid"])

    def objetivo(params):
        return np.mean([rmse(y[va_], modelo(params).fit(X[tr_], y[tr_], est[tr_], t[tr_]).predict(X[va_], est[va_]))
                        for tr_, va_ in internos])

    t0 = time.perf_counter()
    res = aleatoria(ESPACIO, objetivo, cfg["presupuesto"], semilla=SEED)
    t_busqueda = time.perf_counter() - t0
    t1 = time.perf_counter()
    m = modelo(res["mejor_params"]).fit(X[tr_ext], y[tr_ext], est[tr_ext], t[tr_ext])
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = m.predict(X_ev, est_ev).astype(np.float32)
    t_pred = time.perf_counter() - t2
    np.save(carpeta / f"MERF__{pliegue}.npy", pred)
    res.update({"pliegue": pliegue, "semilla": SEED, "iteraciones_EM": ITERACIONES, "segundos_busqueda": t_busqueda,
                "segundos_ajuste_final": t_ajuste, "segundos_prediccion": t_pred,
                "filas_ajuste_final": int(len(tr_ext)), "filas_evaluadas": int(len(y_ev)), "rmse": rmse(y_ev, pred),
                "sigma2": m.sigma2, "D_diag": np.diag(m.D).tolist()})
    tmp = salida.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(salida)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    cfg = ({"prueba": True, "presupuesto": 2, "n_ajuste": 3000, "n_valid": 1000} if args.prueba else
           {"prueba": False, "presupuesto": 40, "n_ajuste": 40000, "n_valid": 10000})
    carpeta = RESULTADOS / ("merf_anidado_prueba" if args.prueba else "merf_anidado")
    carpeta.mkdir(exist_ok=True)
    TR, TE, F = cargar()
    X, y, est, t = TR[F].to_numpy(np.float32), TR.y.to_numpy(), TR.est.to_numpy(), TR.t.to_numpy()
    externos = pliegues_externos(TR)[:1] if args.prueba else pliegues_externos(TR)

    filas, ys, ts, p_m, p_rf = [], [], [], [], []
    for n, _, va in externos:
        t0 = time.time()
        r = pliegue_merf(n, cfg, carpeta)
        rf = np.load(RESULTADOS / "corridas" / f"{RF}__{n}.npy")
        pm = np.load(carpeta / f"MERF__{n}.npy")
        filas.append({"pliegue": n, "RMSE MERF": r["rmse"], "RMSE RF": rmse(y[va], rf),
                      **{k: r["mejor_params"][k] for k in ["efectos", "ventana"]}})
        ys.append(y[va]), ts.append(t[va]), p_m.append(pm), p_rf.append(rf)
        print(n, f"{(time.time() - t0) / 60:.1f} min", filas[-1], flush=True)
    R = pd.DataFrame(filas).set_index("pliegue")
    print(R.round(4).to_string(), flush=True)

    # regla de decisión fijada de antemano (ver el docstring)
    yy, tt = np.concatenate(ys), np.concatenate(ts)
    L = ev.perdida_diaria(tt, yy, {"MERF": np.concatenate(p_m), "RF": np.concatenate(p_rf)})
    dm = ev.diebold_mariano(L.MERF, L.RF)
    gana = bool(dm["diferencia_media"] < 0 and dm["p_valor"] < 0.05)
    decision = {"RMSE 2024 MERF": rmse(yy, np.concatenate(p_m)), "RMSE 2024 RF": rmse(yy, np.concatenate(p_rf)),
                "DM-HLN vs RF": dm["DM_HLN"], "p": dm["p_valor"], "evaluar_2025": gana and not args.prueba}
    (carpeta / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Decisión:", decision, flush=True)

    if decision["evaluar_2025"]:
        r = pliegue_merf("final", cfg, carpeta)
        pm = np.load(carpeta / "MERF__final.npy").astype(float)
        tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
        OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)
        pred = {"MERF": pm}
        for m_, o_ in OPT.items():
            nom = f"{m_}__{o_}__final".replace(" ", "_").replace("(", "").replace(")", "")
            pred[m_] = np.load(RESULTADOS / "corridas" / f"{nom}.npy").astype(float)
        y_te = TE.y.to_numpy()
        L = ev.perdida_diaria(TE.t, y_te, pred)
        from statsmodels.stats.multitest import multipletests
        cmp = pd.DataFrame([{"rival": m_, "RMSE rival": rmse(y_te, pred[m_]),
                             **{k: v for k, v in ev.diebold_mariano(L.MERF, L[m_]).items()}} for m_ in OPT]).set_index("rival")
        cmp["p (Holm)"] = multipletests(cmp.p_valor, method="holm")[1]
        cmp.to_csv(carpeta / "2025_pruebas.csv")
        print(f"\n2025: RMSE MERF {rmse(y_te, pm):.4f}\n{cmp.round(4).to_string()}", flush=True)
        print("MCS:\n", ev.mcs(L).round(4).to_string(), flush=True)

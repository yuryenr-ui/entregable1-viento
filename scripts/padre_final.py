"""Modelo original final: reentrenamiento con 2020–2024, artefactos serializados y SHAP frente a Random Forest.

Uso (desde la carpeta entregable1, después de ``ensamble_padre.py``):
    python scripts/padre_final.py            # todo (≈ 30 min con 10 hilos)
    python scripts/padre_final.py --hilos 16

1. Ajusta ``EnsamblePadre`` (e3/modelo_original.py) con todo el entrenamiento y los hiperparámetros de la búsqueda
   final, y comprueba que su pronóstico de 2025 es el mismo que el guardado por ``ensamble_padre.py``.
2. Serializa con joblib (``compress=3``) en ``artifacts/`` los componentes livianos (XGBoost, SVR y PADRE-XGB),
   cada uno con su ``metadata.json`` (versiones, semilla, hiperparámetros, esquema de columnas con rangos y
   métricas del bucle externo), y verifica la ida y vuelta: el modelo cargado da exactamente las mismas
   predicciones que el modelo en memoria sobre un conjunto fijo. Random Forest y PADRE-RF no se serializan: con
   232 árboles de profundidad 21 pesan varios GB (PADRE-RF se entrena con cuatro veces más filas); se reconstruyen
   con este script.
3. SHAP (``PermutationExplainer``) del ensamble y de Random Forest sobre las mismas 200 filas de 2025 (40 de cada
   una de las 5 estaciones con más datos). La estación no se explica: en cada estación el fondo son 50 filas de
   entrenamiento de esa estación y la función a explicar mantiene fija la columna de estación.

Resultados en artifacts/ y resultados/e3/ensamble_padre/shap_*.csv.
"""
import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
warnings.filterwarnings("ignore")
from e3 import SEED                                                                         # noqa: E402
from e3.modelo_original import EnsamblePadre, PadreEstacionRegresor                          # noqa: E402
from e3.modelos import construir                                                             # noqa: E402
from ensamble_padre import ENSAMBLE, OPT, RESULTADOS, SALIDA, TE, TR, F, nombre, co          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ARTEFACTOS = AQUI.parent / "artifacts"
CORTOS = {"Random Forest": "RF", "XGBoost": "XGB"}


def params(m):
    return json.loads((RESULTADOS / "corridas" / f"{nombre(m, OPT[m], 'final')}.json").read_text(encoding="utf-8"))


def con_estacion(D):
    return np.column_stack([D.est.to_numpy().astype(float), D[F].to_numpy(np.float32)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=10)
    args = ap.parse_args()
    Xtr, Xte, ytr, yte = con_estacion(TR), con_estacion(TE), TR.y.to_numpy(), TE.y.to_numpy()
    comp = [(m, construir(m, params(m)["mejor_params"], n_jobs=args.hilos)) for m in ["Random Forest", "XGBoost", "SVR"]]
    comp += [(f"PADRE-{CORTOS[m]}", PadreEstacionRegresor(construir(m, params(m)["mejor_params"], n_jobs=args.hilos),
                                                           pares=4, anclas=20, coords=co))
             for m in ["XGBoost", "Random Forest"]]

    # 1. ajuste final y verificación contra la caché
    t0 = time.time()
    ens = EnsamblePadre(comp).fit(Xtr, ytr)
    t_aj = time.time() - t0
    t0 = time.time()
    C = ens.predict_componentes(Xte)
    t_pr = time.time() - t0
    guardado = np.mean([np.load(RESULTADOS / "corridas" / f"{nombre(m, OPT[m], 'final')}.npy") for m in
                        ["Random Forest", "XGBoost", "SVR"]] + [np.load(SALIDA / f"PADRE_{k}__final.npy") for k in
                                                                ["XGB", "RF"]], 0)
    dif = float(np.max(np.abs(C.mean(1).to_numpy() - guardado)))
    rm = float(np.sqrt(np.mean((C.mean(1).to_numpy() - yte) ** 2)))
    print(f"ajuste {t_aj:.0f} s | predicción 2025 {t_pr:.0f} s | RMSE 2025 {rm:.4f} | máx |dif| con la caché {dif:.2e}",
          flush=True)

    # 2. artefactos livianos con metadata y prueba de ida y vuelta
    ARTEFACTOS.mkdir(exist_ok=True)
    import sklearn
    import xgboost
    versiones = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}
    resumen = pd.read_csv(SALIDA / "validacion_resumen.csv", index_col=0) if (SALIDA / "validacion_resumen.csv").exists() else None
    esquema = [{"columna": "est", "tipo": "código numérico de estación IDEAM", "rango": "estaciones del catálogo"}] + [
        {"columna": c, "tipo": "float32", "min": float(TR[c].min()), "max": float(TR[c].max())} for c in F]
    fijo = Xte[:5000]
    ida_vuelta = {}
    for nombre_c, modelo in ens.estimadores_:
        if nombre_c in ("Random Forest", "PADRE-RF"):
            continue
        archivo = ARTEFACTOS / f"{nombre_c.replace(' ', '_')}.joblib"
        joblib.dump(modelo, archivo, compress=3)
        entrada = fijo if isinstance(modelo, PadreEstacionRegresor) else fijo[:, 1:]
        igual = bool(np.array_equal(modelo.predict(entrada), joblib.load(archivo).predict(entrada)))
        ida_vuelta[nombre_c] = igual
        base = nombre_c.replace("PADRE-", "")
        motor = {"XGB": "XGBoost"}.get(base, base)
        meta = {"modelo": nombre_c, "semilla": SEED, "versiones": versiones,
                "hiperparametros": params(motor)["mejor_params"] | ({"pares": 4, "anclas": 20, "min_filas": 500}
                                                                      if nombre_c.startswith("PADRE") else {}),
                "esquema_entrada": esquema if nombre_c.startswith("PADRE") else esquema[1:],
                "metricas_bucle_externo": None if resumen is None or nombre_c not in resumen.index else
                resumen.loc[nombre_c, ["RMSE bucle externo (media)", "RMSE bucle externo (DE)"]].to_dict(),
                "tamano_MB": round(archivo.stat().st_size / 2 ** 20, 1), "prueba_ida_y_vuelta": igual}
        (ARTEFACTOS / f"{nombre_c.replace(' ', '_')}.metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(nombre_c, f"{meta['tamano_MB']} MB", "ida y vuelta:", igual, flush=True)

    # 3. SHAP del ensamble y de Random Forest
    import shap
    rng = np.random.default_rng(SEED)
    rf = dict(ens.estimadores_)["Random Forest"]
    top = TE.est.value_counts().index[:5]
    filas_shap = []
    t0 = time.time()
    for e in top:
        idx_te = rng.choice(np.flatnonzero(TE.est.to_numpy() == e), 40, replace=False)
        fondo = TR[F].to_numpy(np.float32)[rng.choice(np.flatnonzero(TR.est.to_numpy() == e), 50, replace=False)]
        cod = float(e)
        f_ens = lambda Z, cod=cod: ens.predict(np.column_stack([np.full(len(Z), cod), Z]))
        for nom_m, f in [(ENSAMBLE, f_ens), ("Random Forest", rf.predict)]:
            exp = shap.PermutationExplainer(f, shap.maskers.Independent(fondo), seed=SEED)
            sv = exp(Xte[idx_te][:, 1:])
            for k, i in enumerate(idx_te):
                filas_shap.append({"modelo": nom_m, "est": e, "fila_2025": int(i), "base": float(sv.base_values[k]),
                                   **dict(zip(F, sv.values[k]))})
        print(f"SHAP estación {e}: {time.time() - t0:.0f} s", flush=True)
    S = pd.DataFrame(filas_shap)
    S.to_csv(SALIDA / "shap_valores.csv", index=False)
    imp = S.groupby("modelo")[F].apply(lambda g: g.abs().mean()).T
    imp["rango ensamble"] = imp[ENSAMBLE].rank(ascending=False)
    imp["rango RF"] = imp["Random Forest"].rank(ascending=False)
    imp.sort_values(ENSAMBLE, ascending=False).to_csv(SALIDA / "shap_importancia.csv")
    from scipy.stats import spearmanr
    rho = spearmanr(imp[ENSAMBLE], imp["Random Forest"]).correlation
    (SALIDA / "final_resumen.json").write_text(json.dumps({
        "segundos_ajuste": t_aj, "segundos_prediccion_2025": t_pr, "rmse_2025": rm, "max_dif_cache": dif,
        "ida_y_vuelta": ida_vuelta, "spearman_importancia_shap": float(rho)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(imp.sort_values(ENSAMBLE, ascending=False).round(4).head(12).to_string())
    print(f"Spearman entre las importancias SHAP del ensamble y de RF: {rho:.3f}")

"""Variante sin modelos del curso: promedio simple de PADRE-XGB y PADRE-RF ("PADRE (2)").

Uso (desde la carpeta entregable1):
    python scripts/padre_solo.py

Motivo: responder a la objeción de que el ensamble de 5 incluye modelos del curso. Los dos PADRE son los mismos
del ensamble (4 parejas, 20 anclas, motores con los hiperparámetros del Entregable 2) y están en caché en
resultados/e3/ensamble_padre/; aquí solo se promedian, sin pesos ajustados.

Regla fijada ANTES de ver los resultados (19/10/2026): en los 4 trimestres de 2024 se compara PADRE (2) con cada
uno de los 7 modelos del curso mediante Diebold-Mariano (HLN) sobre la pérdida diaria, con corrección de Holm.
Solo si tiene menor pérdida que los 7 y p (Holm) < 0.05 en todos, se evalúa una vez en 2025. Si no, se reporta
como intento y 2025 no se usa. La decisión queda en resultados/e3/padre_solo/decision.json.
"""
import json
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
CACHE = RESULTADOS / "ensamble_padre"
SALIDA = RESULTADOS / "padre_solo"
SALIDA.mkdir(exist_ok=True)
NOM = "PADRE (2)"
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def nombre(modelo, optimizador, pliegue):
    return f"{modelo}__{optimizador}__{pliegue}".replace(" ", "_").replace("(", "").replace(")", "")


def periodo(partes, y_de, t_de):
    ys, ts, preds = [], [], {}
    for n, idx in partes:
        p = {m: np.load(RESULTADOS / "corridas" / f"{nombre(m, o, n)}.npy").astype(float) for m, o in OPT.items()}
        p[NOM] = np.mean([np.load(CACHE / f"PADRE_{c}__{n}.npy") for c in ["XGB", "RF"]], 0)
        ys.append(y_de(idx)), ts.append(t_de(idx))
        for k, v in p.items():
            preds.setdefault(k, []).append(v)
    return np.concatenate(ys), np.concatenate(ts), {k: np.concatenate(v) for k, v in preds.items()}


def pruebas(y_ev, t, preds):
    L = ev.perdida_diaria(t, y_ev, preds)
    cmp = pd.DataFrame([{"rival": m, "RMSE rival": rmse(y_ev, preds[m]), **ev.diebold_mariano(L[NOM], L[m])}
                        for m in OPT]).set_index("rival")
    cmp["p (Holm)"] = multipletests(cmp.p_valor, method="holm")[1]
    return cmp, L


if __name__ == "__main__":
    TR, TE, F = cargar()
    tm = pd.read_csv(RESULTADOS / "tabla_maestra_resumen.csv")
    OPT = dict(tm[tm.elegido].set_index("modelo").optimizador)

    yy, tt, P = periodo([(n, va) for n, _, va in pliegues_externos(TR)],
                        lambda i: TR.y.to_numpy()[i], lambda i: TR.t.to_numpy()[i])
    cmp, _ = pruebas(yy, tt, P)
    cmp.to_csv(SALIDA / "2024_pruebas.csv")
    print(f"2024: RMSE {NOM} {rmse(yy, P[NOM]):.4f}\n{cmp.round(4).to_string()}", flush=True)
    gana = bool((cmp.diferencia_media < 0).all() and (cmp["p (Holm)"] < 0.05).all())
    decision = {"RMSE 2024": rmse(yy, P[NOM]), "gana a los 7 en 2024": gana, "evaluar_2025": gana}
    (SALIDA / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Decisión:", decision, flush=True)

    if gana:
        y5, t5, P5 = periodo([("final", None)], lambda i: TE.y.to_numpy(), lambda i: TE.t.to_numpy())
        cmp5, L5 = pruebas(y5, t5, P5)
        cmp5.to_csv(SALIDA / "2025_pruebas.csv")
        print(f"\n2025: RMSE {NOM} {rmse(y5, P5[NOM]):.4f}\n{cmp5.round(4).to_string()}", flush=True)
        print("MCS:\n", ev.mcs(L5).round(4).to_string(), flush=True)

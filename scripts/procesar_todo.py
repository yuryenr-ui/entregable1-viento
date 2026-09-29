"""Procesa todas las descargas de la carpeta 'DATA SET PROPIO' y arma el panel multivariado.

Uso: python procesar_todo.py "C:/Users/USER/Downloads/DATA SET PROPIO" [--causal]
Busca los CSV de cada variable en su subcarpeta; basta con agregar archivos nuevos y volver a ejecutar.
Sin opciones aplica la limpieza original (retrospectiva, procesar_variable.py). Con --causal aplica la
limpieza causal (procesar_variable_causal.py), que solo usa información anterior a cada lectura, y
escribe panel_multivariado_causal.csv. Los resultados se escriben en la carpeta desde la que se ejecuta.
"""
import subprocess
import sys
from pathlib import Path

raiz = Path(sys.argv[1])
CAUSAL = "--causal" in sys.argv
aqui = Path(__file__).resolve().parent  # los demás scripts están junto a este
carpetas = {"velocidad": "VELOCIDAD DEL VIENTO", "direccion": "DIRECCION DEL VIENTO",
            "temperatura": "TEMPERATURA", "presion": "PRESION"}

catalogo = next((raiz / "CATALOGO estaciones").glob("*.csv"))
for var, sub in carpetas.items():
    archivos = sorted(str(f) for f in (raiz / sub).glob("*.csv"))
    if not archivos:
        print(f"{var}: sin archivos")
        continue
    suf = "_causal" if CAUSAL else ""
    print(f"{var}: {len(archivos)} archivos → panel_{var}{suf}.csv (log_{var}{suf}.txt)", flush=True)
    comando = ([sys.executable, str(aqui / "procesar_variable_causal.py"), var, str(catalogo), *archivos] if CAUSAL
               else [sys.executable, str(aqui / "procesar_variable.py"), var, *archivos])
    with open(f"log_{var}{suf}.txt", "w", encoding="utf-8") as log:
        subprocess.run(comando, stdout=log, stderr=log, check=True)

subprocess.run([sys.executable, str(aqui / "unir_panel.py"), str(catalogo)] + (["--causal"] if CAUSAL else []), check=True)

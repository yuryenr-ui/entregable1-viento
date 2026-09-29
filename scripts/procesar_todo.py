"""Procesa todas las descargas de la carpeta 'DATA SET PROPIO' y arma el panel multivariado.

Uso: python procesar_todo.py "C:/Users/USER/Downloads/DATA SET PROPIO"
Busca los CSV de cada variable en su subcarpeta; basta con agregar archivos nuevos y volver a ejecutar.
"""
import subprocess
import sys
from pathlib import Path

raiz = Path(sys.argv[1])
aqui = Path(__file__).resolve().parent  # los demás scripts están junto a este
carpetas = {"velocidad": "VELOCIDAD DEL VIENTO", "direccion": "DIRECCION DEL VIENTO",
            "temperatura": "TEMPERATURA", "presion": "PRESION"}

for var, sub in carpetas.items():
    archivos = sorted(str(f) for f in (raiz / sub).glob("*.csv"))
    if not archivos:
        print(f"{var}: sin archivos")
        continue
    print(f"{var}: {len(archivos)} archivos → panel_{var}.csv (log_{var}.txt)")
    with open(f"log_{var}.txt", "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(aqui / "procesar_variable.py"), var, *archivos], stdout=log, stderr=log, check=True)

catalogo = next((raiz / "CATALOGO estaciones").glob("*.csv"))
subprocess.run([sys.executable, str(aqui / "unir_panel.py"), str(catalogo)], check=True)

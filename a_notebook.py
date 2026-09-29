"""Convierte pruebas_nasa_power.py (formato de celdas '# %%') en un notebook .ipynb."""
import re
import sys
from pathlib import Path

import nbformat

src = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

for bloque in re.split(r"^# %%", src, flags=re.M)[1:]:
    cabecera, _, cuerpo = bloque.partition("\n")
    if "[markdown]" in cabecera:
        texto = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in cuerpo.strip().splitlines())
        nb.cells.append(nbformat.v4.new_markdown_cell(texto))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(cuerpo.strip()))

nbformat.write(nb, out)
print(f"{out}: {len(nb.cells)} celdas")

import sys
from pathlib import Path

from cv_metadata import get_top_female_speakers

DATA_DIR = Path("./data/cv-raw")

if not DATA_DIR.exists():
    print(f"Error: corpus no encontrado en {DATA_DIR}.")
    print("Ejecuta primero: task descargar-cv")
    sys.exit(1)

print("\n--- Analizando hablantes de Common Voice Spanish ---")
df = get_top_female_speakers(DATA_DIR, n=10)

if df.empty:
    print("Error: no se encontraron hablantes femeninas en el corpus.")
    sys.exit(1)

print("\nTop hablantes femeninas en Common Voice Spanish (por minutos estimados):\n")
print(f"{'#':<4} {'gender':<16} {'clips':>8} {'minutos':>10}  client_id")
print("-" * 46)
for i, row in df.iterrows():
    print(
        f"{i + 1:<4} {row['gender']:<16} {int(row['clips']):>8} {row['minutes']:>10.1f}  {row['client_id']}"
    )

best = df.iloc[0]["client_id"]
print("\nPara exportar la hablante con más audio:")
print(f"  uv run exportar_cv.py --speaker {best}")

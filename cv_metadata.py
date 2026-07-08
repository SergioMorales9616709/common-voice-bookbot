from pathlib import Path

import pandas as pd

SECONDS_PER_CLIP = 5.0
_REQUIRED_COLS = {"client_id", "path", "gender", "sentence"}


def find_clips_tsv(data_dir: Path) -> Path:
    """Return the first TSV in data_dir that has the required CV columns."""
    for tsv in sorted(data_dir.rglob("*.tsv")):
        try:
            with tsv.open(encoding="utf-8") as f:
                header = f.readline()
            if _REQUIRED_COLS.issubset(set(header.rstrip("\n").split("\t"))):
                return tsv
        except OSError:
            continue
    raise FileNotFoundError(
        f"No se encontró un TSV de clips de Common Voice en {data_dir}. Ejecuta: task descargar-cv"
    )


def get_top_female_speakers(data_dir: Path, n: int = 20) -> pd.DataFrame:
    """Return top-n female speakers sorted by estimated minutes."""
    tsv_path = find_clips_tsv(data_dir)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, usecols=["client_id", "gender"])
    females = df[df["gender"] == "female"]
    counts = females.groupby("client_id", as_index=False).size().rename(columns={"size": "clips"})
    counts["clips"] = counts["clips"].astype(int)
    counts["minutes"] = counts["clips"] * SECONDS_PER_CLIP / 60
    return counts.sort_values("minutes", ascending=False).head(n).reset_index(drop=True)

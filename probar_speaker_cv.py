"""
Preview tool: export N sample clips from a Common Voice speaker for quality evaluation.

Unlike probar_speaker.py (MLS), Common Voice clips are already local files once
./data/cv-raw/ is extracted — no need to stream shards from Hugging Face. This just
reads the clips TSV, grabs the speaker's first N rows, and processes them through the
same audio_utils pipeline used by exportar_cv.py (decode -> resample 22050 -> loudness
normalize -> PCM_16 wav), so what you hear in the preview matches exactly what a real
export would produce.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from audio_utils import process_clip, save_wav
from cv_metadata import find_clips_tsv
from exportar_dataset import decode_audio_bytes


def preview_speaker(speaker_id: str, data_dir: Path, n: int) -> None:
    tsv_path = find_clips_tsv(data_dir)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    rows = df[df["client_id"] == speaker_id].head(n).reset_index(drop=True)

    if rows.empty:
        print(f"\nError: no se encontraron clips para client_id='{speaker_id}'.")
        print("Ejecuta: uv run analizar_cv.py  — para ver client_ids válidos.")
        sys.exit(1)

    clips_dir = tsv_path.parent / "clips"
    out_dir = Path("./preview") / speaker_id[:8]
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, row in rows.iterrows():
        audio_path = clips_dir / row["path"]
        if not audio_path.exists():
            print(f"  Warning: falta {audio_path.name}")
            continue

        try:
            audio, sr = decode_audio_bytes(audio_path.read_bytes())
            processed, target_sr = process_clip(audio, sr)
        except Exception as e:
            print(f"  Warning: clip omitido (decode error: {e})")
            continue

        out_path = out_dir / f"{i + 1:02d}.wav"
        save_wav(processed, target_sr, out_path)
        duration = len(processed) / target_sr
        sentence = str(row.get("sentence", "")).strip()[:70]
        print(f'  [{i + 1}] {out_path.name}  {duration:.1f}s  "{sentence}"')
        saved += 1

    if saved == 0:
        print("\nNo se pudo procesar ningún clip.")
        sys.exit(1)

    print(f"\n{saved} clips guardados en: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta N clips de muestra de un hablante de Common Voice para evaluar calidad"
    )
    parser.add_argument("--speaker", required=True, help="client_id de Common Voice")
    parser.add_argument("--n", type=int, default=5, help="Número de clips (default: 5)")
    parser.add_argument(
        "--data-dir",
        default="./data/cv-raw",
        help="Directorio del corpus descargado (default: ./data/cv-raw)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} no existe. Ejecuta: task descargar-cv")
        sys.exit(1)

    preview_speaker(args.speaker, data_dir, args.n)


if __name__ == "__main__":
    main()

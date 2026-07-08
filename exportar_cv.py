import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from audio_utils import process_clip, save_wav
from cv_metadata import find_clips_tsv
from exportar_dataset import decode_audio_bytes, write_metadata_csv

load_dotenv()

MIN_DURATION_SECONDS = 0.5


def build_wav_filename(client_id: str, index: int) -> str:
    return f"{client_id[:8]}_{index:04d}.wav"


def export_speaker(client_id: str, data_dir: Path, output_dir: Path) -> None:
    tsv_path = find_clips_tsv(data_dir)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    speaker_rows = df[df["client_id"] == client_id].reset_index(drop=True)

    if speaker_rows.empty:
        print(f"\nError: no se encontraron clips para client_id='{client_id}'.")
        print("Ejecuta analizar_cv.py para ver client_ids válidos.")
        sys.exit(1)

    clips_dir = tsv_path.parent / "clips"
    wavs_dir = output_dir / "wavs"
    all_entries: list[tuple[str, str]] = []
    clip_index = 1
    total_skipped = 0

    for _, row in tqdm(speaker_rows.iterrows(), total=len(speaker_rows), desc="clips", unit="clip"):
        audio_path = clips_dir / row["path"]
        if not audio_path.exists():
            total_skipped += 1
            continue

        try:
            audio_array, source_sr = decode_audio_bytes(audio_path.read_bytes())
        except Exception as e:
            print(f"  Warning: decode error ({e})")
            total_skipped += 1
            continue

        if len(audio_array) / source_sr < MIN_DURATION_SECONDS:
            total_skipped += 1
            continue

        try:
            processed, target_sr = process_clip(audio_array, source_sr)
            wav_name = build_wav_filename(client_id, clip_index)
            save_wav(processed, target_sr, wavs_dir / wav_name)
        except Exception as e:
            print(f"  Warning: process error ({e})")
            total_skipped += 1
            continue

        sentence = str(row.get("sentence", "")).strip()
        all_entries.append((f"wavs/{wav_name}", sentence))
        clip_index += 1

    if not all_entries:
        print(f"\nError: no se exportaron clips para client_id='{client_id}'.")
        sys.exit(1)

    write_metadata_csv(all_entries, output_dir / "metadata.csv")
    print(f"\nExportados : {len(all_entries)} clips")
    print(f"Omitidos   : {total_skipped} clips")
    print(f"Dataset en : {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exportar hablante Common Voice en formato LJSpeech para piper-tts"
    )
    parser.add_argument("--speaker", required=True, help="client_id de Common Voice")
    parser.add_argument(
        "--data-dir",
        default="./data/cv-raw",
        help="Directorio del corpus descargado (default: ./data/cv-raw)",
    )
    parser.add_argument(
        "--output", default=None, help="Directorio de salida (default: ./data/<client_id[:8]>)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} no existe. Ejecuta: task descargar-cv")
        sys.exit(1)

    output = Path(args.output) if args.output else Path("./data") / args.speaker[:8]
    export_speaker(args.speaker, data_dir, output)


if __name__ == "__main__":
    main()

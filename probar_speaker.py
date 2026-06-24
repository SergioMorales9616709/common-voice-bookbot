import argparse
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem

from exportar_dataset import decode_audio_bytes
from audio_utils import process_clip, save_wav

load_dotenv()

HF_BASE = "datasets/facebook/multilingual_librispeech/spanish"
# Check smaller splits first (single files), then train shards
SPLIT_PATTERNS = ["dev-*.parquet", "test-*.parquet", "train-*.parquet"]


def _scan_shard(fs: HfFileSystem, path: str, speaker_id: str) -> list[int]:
    """Read only speaker_id column from a parquet shard. Returns matching row indices."""
    with fs.open(path, "rb") as fh:
        table = pq.read_table(fh, columns=["speaker_id"])
    ids = [str(x) for x in table["speaker_id"].to_pylist()]
    return [i for i, sid in enumerate(ids) if sid == speaker_id]


def _extract_clips(fs: HfFileSystem, path: str, row_indices: list[int]) -> list[dict]:
    """Download a shard and extract specific rows."""
    with fs.open(path, "rb") as fh:
        table = pq.read_table(fh, columns=["audio", "text"])
    return [
        {"audio": table["audio"][i].as_py(), "text": str(table["text"][i].as_py())}
        for i in row_indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga N clips de muestra de un hablante MLS para escuchar antes de exportar"
    )
    parser.add_argument("--speaker", required=True, help="speaker_id de MLS (ej: 1447)")
    parser.add_argument("--n", type=int, default=5, help="Número de clips (default: 5)")
    args = parser.parse_args()

    speaker_id = str(args.speaker)
    out_dir = Path("./preview") / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    saved = 0

    for pattern in SPLIT_PATTERNS:
        if saved >= args.n:
            break

        shards = sorted(fs.glob(f"{HF_BASE}/{pattern}"))
        for shard in shards:
            if saved >= args.n:
                break

            split_name = Path(shard).name.split("-")[0]
            print(f"  Escaneando {Path(shard).name}...", end=" ", flush=True)

            try:
                rows = _scan_shard(fs, shard, speaker_id)
            except Exception as e:
                print(f"error: {e}")
                continue

            if not rows:
                print("sin clips")
                continue

            needed = min(len(rows), args.n - saved)
            print(f"{len(rows)} clips encontrados — descargando {needed}...")

            try:
                clips = _extract_clips(fs, shard, rows[:needed])
            except Exception as e:
                print(f"  Error al descargar shard: {e}")
                continue

            for clip in clips:
                audio_bytes = clip["audio"].get("bytes") if isinstance(clip["audio"], dict) else None
                if not audio_bytes:
                    continue
                try:
                    audio, sr = decode_audio_bytes(audio_bytes)
                    processed, target_sr = process_clip(audio, sr)
                    idx = str(saved + 1).zfill(2)
                    out_path = out_dir / f"{idx}.wav"
                    save_wav(processed, target_sr, out_path)
                    duration = len(processed) / target_sr
                    print(f"  [{saved + 1}] {out_path.name}  {duration:.1f}s  \"{clip['text']}\"")
                    saved += 1
                except Exception as e:
                    print(f"  Warning: clip omitido ({e})")

    if saved == 0:
        print(f"\nNo se encontraron clips para speaker_id='{speaker_id}'.")
        print("Ejecuta: task analizar  — para ver speaker_ids válidos.")
    else:
        print(f"\n{saved} clips guardados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

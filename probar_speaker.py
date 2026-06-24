"""
Preview tool: download N sample clips from an MLS speaker for quality evaluation.

Search order (fastest to slowest):
  1_hours (~4s) → 9_hours (~30s) → dev (~60s) → test (~60s) → train shards (3-5 min each)

Train shards are only scanned if --full-scan is passed, since each takes 3-5 min.
Once downloaded, HuggingFace caches shards locally — subsequent runs are instant.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem

from audio_utils import process_clip, save_wav
from exportar_dataset import decode_audio_bytes

load_dotenv()

HF_BASE = "datasets/facebook/multilingual_librispeech/spanish"

# Search order: small curated files first, large train shards last
FAST_PATTERNS = [
    "1_hours-*.parquet",
    "9_hours-*.parquet",
    "dev-*.parquet",
    "test-*.parquet",
]
TRAIN_PATTERN = "train-*.parquet"


def _load_shard_clips(fs: HfFileSystem, path: str, speaker_id: str, max_clips: int) -> list[dict]:
    """Download a parquet shard and return up to max_clips rows for the speaker."""
    with fs.open(path, "rb") as fh:
        table = pq.read_table(fh, columns=["speaker_id", "audio", "transcript"])
    results = []
    for i in range(len(table)):
        if str(table["speaker_id"][i].as_py()) == speaker_id:
            results.append(
                {
                    "audio": table["audio"][i].as_py(),
                    "transcript": str(table["transcript"][i].as_py()),
                }
            )
            if len(results) >= max_clips:
                break
    return results


def _save_clips(clips: list[dict], out_dir: Path, start_index: int) -> int:
    """Process and save clips. Returns number of clips saved."""
    saved = 0
    for clip in clips:
        audio_bytes = clip["audio"].get("bytes") if isinstance(clip["audio"], dict) else None
        if not audio_bytes:
            continue
        try:
            audio, sr = decode_audio_bytes(audio_bytes)
            processed, target_sr = process_clip(audio, sr)
            idx = str(start_index + saved + 1).zfill(2)
            out_path = out_dir / f"{idx}.wav"
            save_wav(processed, target_sr, out_path)
            duration = len(processed) / target_sr
            print(
                f'  [{start_index + saved + 1}] {out_path.name}  {duration:.1f}s  "{clip["text"][:70]}"'
            )
            saved += 1
        except Exception as e:
            print(f"  Warning: clip omitido ({e})")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga N clips de muestra de un hablante MLS")
    parser.add_argument("--speaker", required=True, help="speaker_id (ej: 1447)")
    parser.add_argument("--n", type=int, default=5, help="Número de clips (default: 5)")
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Escanear también shards de train (3-5 min por shard, cached tras primer uso)",
    )
    args = parser.parse_args()

    speaker_id = str(args.speaker)
    out_dir = Path("./preview") / speaker_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    saved = 0

    print(f"\nBuscando clips del hablante {speaker_id} (primeros {args.n})...")

    # Phase 1: fast files (curated subsets + dev/test)
    for pattern in FAST_PATTERNS:
        if saved >= args.n:
            break
        shards = sorted(fs.glob(f"{HF_BASE}/{pattern}"))
        for shard in shards:
            if saved >= args.n:
                break
            name = Path(shard).name
            print(f"  {name}...", end=" ", flush=True)
            try:
                clips = _load_shard_clips(fs, shard, speaker_id, args.n - saved)
            except Exception as e:
                print(f"error: {e}")
                continue
            if not clips:
                print("sin clips")
                continue
            print(f"{len(clips)} clips")
            saved += _save_clips(clips, out_dir, saved)

    # Phase 2: train shards (slow, opt-in)
    if saved < args.n:
        if not args.full_scan:
            print(
                f"\n  No se encontraron suficientes clips en los archivos rápidos ({saved}/{args.n})."
                f"\n  Para buscar en los shards de entrenamiento (~3-5 min por shard, pero se cachean):"
                f"\n    task probar SPEAKER={speaker_id} -- --full-scan"
            )
        else:
            train_shards = sorted(fs.glob(f"{HF_BASE}/{TRAIN_PATTERN}"))
            total = len(train_shards)
            print(f"\n  Escaneando {total} shards de train (~3-5 min por shard, se cachean)...")
            for i, shard in enumerate(train_shards):
                if saved >= args.n:
                    break
                name = Path(shard).name
                print(f"  [{i + 1}/{total}] {name}...", end=" ", flush=True)
                try:
                    clips = _load_shard_clips(fs, shard, speaker_id, args.n - saved)
                except Exception as e:
                    print(f"error: {e}")
                    continue
                if not clips:
                    print("sin clips")
                    continue
                print(f"{len(clips)} clips")
                saved += _save_clips(clips, out_dir, saved)

    if saved == 0:
        print(f"\nNo se encontraron clips para speaker_id='{speaker_id}'.")
        print("Ejecuta: task analizar  — para ver speaker_ids válidos.")
    else:
        print(f"\n{saved} clips guardados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import io
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from audio_utils import process_clip, save_wav

load_dotenv()

HF_BASE = "datasets/facebook/multilingual_librispeech/spanish"
SPLIT_PATTERNS = {
    "train": "train-*.parquet",
    "dev":   "dev-*.parquet",
    "test":  "test-*.parquet",
}
MIN_DURATION_SECONDS = 0.5


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode raw audio bytes (any format including Opus) to float32 mono array."""
    container = av.open(io.BytesIO(audio_bytes), metadata_errors="ignore")
    stream = next(s for s in container.streams if s.type == "audio")
    sr = stream.sample_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray()[0])
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray()[0])
    container.close()
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio.astype(np.float32), sr


def build_wav_filename(speaker_id: str, index: int) -> str:
    return f"{speaker_id}_{index:04d}.wav"


def write_metadata_csv(entries: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for filename, text in entries:
            f.write(f"{filename}|{text}\n")


def _process_shard(
    fs: HfFileSystem,
    shard_path: str,
    speaker_id: str,
    wavs_dir: Path,
    clip_index: int,
) -> tuple[list[tuple[str, str]], int, int]:
    """Download one shard, process all clips for the speaker. Returns (entries, next_index, skipped)."""
    with fs.open(shard_path, "rb") as fh:
        table = pq.read_table(fh, columns=["speaker_id", "audio", "transcript"])

    entries: list[tuple[str, str]] = []
    skipped = 0

    rows = [i for i in range(len(table)) if str(table["speaker_id"][i].as_py()) == speaker_id]
    for i in tqdm(rows, desc=f"    clips", leave=False, unit="clip"):
        audio_raw = table["audio"][i].as_py()
        audio_bytes = audio_raw.get("bytes") if isinstance(audio_raw, dict) else None
        if not audio_bytes:
            skipped += 1
            continue

        try:
            audio_array, source_sr = decode_audio_bytes(audio_bytes)
        except Exception as e:
            print(f"      Warning: decode error ({e})")
            skipped += 1
            continue

        duration = len(audio_array) / source_sr
        if duration < MIN_DURATION_SECONDS:
            skipped += 1
            continue

        try:
            processed, target_sr = process_clip(audio_array, source_sr)
            wav_name = build_wav_filename(speaker_id, clip_index)
            save_wav(processed, target_sr, wavs_dir / wav_name)
        except Exception as e:
            print(f"      Warning: process error ({e})")
            skipped += 1
            continue

        entries.append((f"wavs/{wav_name}", str(table["transcript"][i].as_py())))
        clip_index += 1

    return entries, clip_index, skipped


def export_speaker(speaker_id: str, output_dir: Path) -> None:
    wavs_dir = output_dir / "wavs"
    fs = HfFileSystem()

    all_entries: list[tuple[str, str]] = []
    clip_index = 1
    total_skipped = 0

    for split, pattern in SPLIT_PATTERNS.items():
        shards = sorted(fs.glob(f"{HF_BASE}/{pattern}"))
        if not shards:
            continue
        print(f"\n  Split '{split}': {len(shards)} shard(s)")
        for idx, shard in enumerate(shards, 1):
            print(f"  [{idx}/{len(shards)}] Descargando {Path(shard).name}...", flush=True)
            try:
                entries, clip_index, skipped = _process_shard(
                    fs, shard, speaker_id, wavs_dir, clip_index
                )
            except Exception as e:
                print(f"      Error en shard: {e}")
                continue
            all_entries.extend(entries)
            total_skipped += skipped
            if entries:
                print(f"      +{len(entries)} clips (total: {len(all_entries)})")

    if not all_entries:
        print(f"\nError: no se encontraron clips para speaker_id='{speaker_id}'.")
        print("Ejecuta analizar_mls.py para ver speaker_ids válidos.")
        sys.exit(1)

    write_metadata_csv(all_entries, output_dir / "metadata.csv")
    print(f"\nExportados : {len(all_entries)} clips")
    print(f"Omitidos   : {total_skipped} clips")
    print(f"Dataset en : {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exportar hablante MLS en formato LJSpeech para piper-tts"
    )
    parser.add_argument("--speaker", required=True, help="speaker_id de MLS (ej: 2138)")
    parser.add_argument(
        "--output", default=None,
        help="Directorio de salida (default: ./data/<speaker_id>)"
    )
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path("./data") / args.speaker
    export_speaker(args.speaker, output)


if __name__ == "__main__":
    main()

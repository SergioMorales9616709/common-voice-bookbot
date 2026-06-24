import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from audio_utils import process_clip, save_wav

MIN_DURATION_SECONDS = 0.5


def build_wav_filename(speaker_id: str, index: int) -> str:
    return f"{speaker_id}_{index:04d}.wav"


def write_metadata_csv(entries: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for filename, text in entries:
            f.write(f"{filename}|{text}\n")


def export_speaker(speaker_id: str, output_dir: Path) -> None:
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_entries: list[tuple[str, str]] = []
    clip_index = 1
    skipped = 0

    for split in ["train", "dev", "test"]:
        try:
            ds = load_dataset(
                "facebook/multilingual_librispeech",
                "spanish",
                split=split,
                streaming=True,
                trust_remote_code=False,
            )
        except Exception as e:
            print(f"  Skipping split '{split}': {e}")
            continue

        speaker_clips = ds.filter(lambda x: x["speaker_id"] == speaker_id)

        for sample in tqdm(speaker_clips, desc=f"  {split}"):
            audio_array = sample["audio"]["array"]
            source_sr = sample["audio"]["sampling_rate"]
            duration = len(audio_array) / source_sr

            if duration < MIN_DURATION_SECONDS:
                skipped += 1
                continue

            try:
                processed, target_sr = process_clip(audio_array, source_sr)
            except Exception as e:
                print(f"  Warning: omitiendo clip (error: {e})")
                skipped += 1
                continue

            wav_name = build_wav_filename(speaker_id, clip_index)
            save_wav(processed, target_sr, wavs_dir / wav_name)
            metadata_entries.append((f"wavs/{wav_name}", sample["text"]))
            clip_index += 1

    if not metadata_entries:
        print(f"\nError: no se encontraron clips para speaker_id='{speaker_id}'.")
        print("Ejecuta analizar_mls.py para ver speaker_ids válidos.")
        sys.exit(1)

    write_metadata_csv(metadata_entries, output_dir / "metadata.csv")
    print(f"\nExportados : {len(metadata_entries)} clips")
    print(f"Omitidos   : {skipped} clips")
    print(f"Dataset en : {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exportar hablante MLS en formato LJSpeech para piper-tts"
    )
    parser.add_argument("--speaker", required=True, help="speaker_id de MLS (ej: 2138)")
    parser.add_argument(
        "--output", default="./data/dataset", help="Directorio de salida (default: ./data/dataset)"
    )
    args = parser.parse_args()
    export_speaker(args.speaker, Path(args.output))


if __name__ == "__main__":
    main()

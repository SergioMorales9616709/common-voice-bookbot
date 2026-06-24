import argparse
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset, Audio
from tqdm import tqdm

from exportar_dataset import decode_audio_bytes
from audio_utils import process_clip, save_wav

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga N clips de un hablante MLS para previsualización"
    )
    parser.add_argument("--speaker", required=True, help="speaker_id de MLS (ej: 1447)")
    parser.add_argument("--n", type=int, default=5, help="Número de clips a descargar (default: 5)")
    args = parser.parse_args()

    out_dir = Path("./preview") / args.speaker
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuscando clips del hablante {args.speaker} (primeros {args.n})...\n")
    saved = 0

    for split in ["train", "dev", "test"]:
        if saved >= args.n:
            break
        try:
            ds = load_dataset(
                "facebook/multilingual_librispeech",
                "spanish",
                split=split,
                streaming=True,
                trust_remote_code=False,
            )
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"  Skipping split '{split}': {e}")
            continue

        clips = ds.filter(lambda x, sid=args.speaker: str(x["speaker_id"]) == str(sid))

        for sample in tqdm(clips, desc=f"  {split}", leave=False):
            if saved >= args.n:
                break
            try:
                audio, sr = decode_audio_bytes(sample["audio"]["bytes"])
                processed, target_sr = process_clip(audio, sr)
                out_path = out_dir / f"{saved + 1:02d}.wav"
                save_wav(processed, target_sr, out_path)
                duration = len(processed) / target_sr
                print(f"  [{saved + 1}] {out_path.name}  {duration:.1f}s  \"{sample['text']}\"")
                saved += 1
            except Exception as e:
                print(f"  Warning: clip omitido ({e})")

    if saved == 0:
        print(f"No se encontraron clips para speaker_id='{args.speaker}'.")
        print("Ejecuta analizar_mls.py para ver speaker_ids válidos.")
    else:
        print(f"\n{saved} clips guardados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

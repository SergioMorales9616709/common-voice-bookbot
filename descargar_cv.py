import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Verified with diagnostic: https://mozilladatacollective.com/api/v1 → 404,
# https://mozilladatacollective.com/api → 200
MDC_API_BASE = "https://mozilladatacollective.com/api"
DATASET_ID = "cmqim2spa00synr071fcp7av0"
OUTPUT_DIR = Path("./data/cv-raw")
PART_FILE = Path("./data/cv-raw.tar.gz.part")
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def get_download_url(api_key: str) -> dict:
    resp = requests.post(
        f"{MDC_API_BASE}/datasets/{DATASET_ID}/download",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


_RETRYABLE = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def download_file(
    url: str,
    dest: Path,
    total_size: int,
    *,
    url_factory: Callable[[], str] | None = None,
    max_retries: int = 10,
) -> None:
    current_url = url
    for attempt in range(max_retries):
        if attempt > 0 and url_factory is not None:
            print("Renovando URL de descarga...")
            current_url = url_factory()

        offset = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        mode = "ab" if offset > 0 else "wb"

        try:
            with requests.get(current_url, headers=headers, stream=True, timeout=(30, 120)) as resp:
                resp.raise_for_status()
                if offset > 0 and resp.status_code != 206:
                    raise RuntimeError(
                        f"El servidor no soporta reanudación (HTTP {resp.status_code} en lugar de 206). "
                        "Elimina el archivo .part y vuelve a intentar."
                    )
                with open(dest, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=offset,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="Descargando",
                    ) as bar:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            f.write(chunk)
                            bar.update(len(chunk))
            return
        except _RETRYABLE:
            if attempt == max_retries - 1:
                raise
            current_offset = dest.stat().st_size if dest.exists() else 0
            wait = min(2**attempt, 60)
            print(
                f"\nConexión interrumpida en {current_offset / 1e9:.2f} GB"
                f" (intento {attempt + 1}/{max_retries}). Reintentando en {wait}s..."
            )
            time.sleep(wait)


def verify_checksum(path: Path, expected: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def extract_archive(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(output_dir, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga Common Voice Scripted Speech 26.0 Spanish a ./data/cv-raw/"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-descarga aunque ya exista ./data/cv-raw/"
    )
    args = parser.parse_args()

    # Si OUTPUT_DIR existe pero NO hay .part, la extracción terminó bien.
    # Si OUTPUT_DIR existe Y hay .part, la extracción fue interrumpida → reanudar.
    extraction_interrupted = OUTPUT_DIR.exists() and PART_FILE.exists()

    if OUTPUT_DIR.exists() and not args.force and not extraction_interrupted:
        print(f"El corpus ya existe en {OUTPUT_DIR}. Usa --force para re-descargar.")
        return

    api_key = os.environ.get("MDC_API_KEY", "")
    if not api_key:
        print("Error: MDC_API_KEY no encontrado en .env")
        sys.exit(1)

    print("Obteniendo URL de descarga...")
    info = get_download_url(api_key)
    url: str = info["downloadUrl"]
    total_size: int = int(info["sizeBytes"])
    checksum: str = info["checksum"]

    PART_FILE.parent.mkdir(parents=True, exist_ok=True)
    already_complete = PART_FILE.exists() and PART_FILE.stat().st_size >= total_size

    if already_complete:
        print("Archivo ya descargado completamente, saltando descarga.")
    elif PART_FILE.exists():
        offset = PART_FILE.stat().st_size
        print(f"Reanudando desde {offset / 1e9:.2f} GB de {total_size / 1e9:.2f} GB...")
        download_file(
            url,
            PART_FILE,
            total_size,
            url_factory=lambda: get_download_url(api_key)["downloadUrl"],
        )
    else:
        print(f"Descargando {total_size / 1e9:.2f} GB...")
        download_file(
            url,
            PART_FILE,
            total_size,
            url_factory=lambda: get_download_url(api_key)["downloadUrl"],
        )

    print("Verificando checksum SHA-256...")
    if not verify_checksum(PART_FILE, checksum):
        PART_FILE.unlink()
        print("Error: checksum no coincide. Archivo eliminado. Vuelve a ejecutar.")
        sys.exit(1)

    if OUTPUT_DIR.exists():
        print(f"Eliminando extracción parcial en {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    print(f"Extrayendo a {OUTPUT_DIR}...")
    extract_archive(PART_FILE, OUTPUT_DIR)
    PART_FILE.unlink()
    print(f"Corpus disponible en {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

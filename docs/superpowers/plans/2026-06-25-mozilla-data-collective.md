# Mozilla Data Collective — Common Voice Local Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken HuggingFace streaming pipeline for Common Voice with a local pipeline: download once from Mozilla Data Collective, then analyze and export from disk.

**Architecture:** `descargar_cv.py` downloads the 48 GB corpus with resume support; `cv_metadata.py` reads the local TSV to rank speakers; `analizar_cv.py` and `exportar_cv.py` are updated to use local files. All audio processing still goes through `audio_utils.py`.

**Tech Stack:** `requests>=2.32.0` (HTTP download with Range resume), `pandas`, `tarfile` (stdlib), `tqdm`, `audio_utils.process_clip`/`save_wav`, `exportar_dataset.decode_audio_bytes`/`write_metadata_csv`

## Global Constraints

- Python 3.13+, `uv` for dependency management
- Dataset ID: `cmqim2spa00synr071fcp7av0` (Common Voice Scripted Speech 26.0 - Spanish, 48.30 GB)
- Corpus extracted to: `./data/cv-raw/` (hardcoded default, not configurable)
- Partial download file: `./data/cv-raw.tar.gz.part`
- `MDC_API_KEY` must be in `.env` (gitignored)
- Do NOT modify: `audio_utils.py`, `mls_metadata.py`, `analizar_mls.py`, `exportar_dataset.py`, `conftest.py`
- Output format: LJSpeech `wavs/<prefix>_NNNN.wav` + `metadata.csv` with `filename|text` rows, no header
- `build_wav_filename` uses `client_id[:8]` as prefix: `f"{client_id[:8]}_{index:04d}.wav"`
- Unit tests run without network or `MDC_API_KEY`; integration tests use `@pytest.mark.integration`
- Run tests: `uv run pytest tests/ -v` (unit) | `uv run pytest tests/ -v -m integration` (integration)
- No git commits — user manages git

---

### Task 1: `descargar_cv.py` — resumable corpus download

**Files:**
- Create: `descargar_cv.py`
- Create: `tests/test_descargar_cv.py`
- Modify: `pyproject.toml` (add `requests>=2.32.0`)

**Interfaces:**
- Produces:
  - `get_download_url(api_key: str) -> dict` — returns `{"downloadUrl": str, "sizeBytes": int, "checksum": str, "filename": str}`
  - `verify_checksum(path: Path, expected: str) -> bool`
  - `extract_archive(archive: Path, output_dir: Path) -> None`
  - `download_file(url: str, dest: Path, total_size: int) -> None`

- [ ] **Step 1: Add `requests` to pyproject.toml**

In `pyproject.toml`, add to the `dependencies` list:
```toml
"requests>=2.32.0",
```
Then run:
```
uv sync
```
Expected: resolves without error.

- [ ] **Step 2: Verify the MDC API base URL**

Run this one-off diagnostic (requires `MDC_API_KEY` in `.env`):
```python
# save as /tmp/check_api.py and run: uv run python /tmp/check_api.py
import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.environ["MDC_API_KEY"]
DATASET_ID = "cmqim2spa00synr071fcp7av0"
for base in [
    "https://mozilladatacollective.com/api/v1",
    "https://api.mozilladatacollective.com/v1",
    "https://mozilladatacollective.com/api",
]:
    r = requests.get(f"{base}/datasets/{DATASET_ID}",
                     headers={"Authorization": f"Bearer {key}"}, timeout=10)
    print(f"{base}: HTTP {r.status_code}")
```
Set `MDC_API_BASE` in `descargar_cv.py` to whichever returns HTTP 200.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_descargar_cv.py`:

```python
import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_tarball(tmp_path: Path, files: dict[str, bytes]) -> Path:
    archive = tmp_path / "corpus.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return archive


def test_verify_checksum_correct(tmp_path: Path):
    content = b"corpus data"
    f = tmp_path / "file.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    from descargar_cv import verify_checksum
    assert verify_checksum(f, expected) is True


def test_verify_checksum_wrong(tmp_path: Path):
    content = b"corpus data"
    f = tmp_path / "file.bin"
    f.write_bytes(content)
    from descargar_cv import verify_checksum
    assert verify_checksum(f, "deadbeef") is False


def test_extract_archive_creates_files(tmp_path: Path):
    archive = _make_tarball(tmp_path, {
        "corpus/es/clips.tsv": b"client_id\tpath\n",
        "corpus/es/clips/001.mp3": b"fakeaudio",
    })
    output_dir = tmp_path / "output"
    from descargar_cv import extract_archive
    extract_archive(archive, output_dir)
    assert (output_dir / "corpus" / "es" / "clips.tsv").exists()
    assert (output_dir / "corpus" / "es" / "clips" / "001.mp3").exists()


def test_get_download_url_returns_dict():
    fake_response = {
        "downloadUrl": "https://storage.example.com/corpus.tar.gz",
        "sizeBytes": 48_000_000_000,
        "checksum": "abc123def456",
        "filename": "common-voice-scripted-speech-26-0-es.tar.gz",
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status = MagicMock()

    with patch("descargar_cv.requests.post", return_value=mock_resp) as mock_post:
        from descargar_cv import get_download_url
        result = get_download_url("my-api-key")

    assert result["downloadUrl"] == fake_response["downloadUrl"]
    assert result["sizeBytes"] == fake_response["sizeBytes"]
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-api-key"


def test_get_download_url_raises_on_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch("descargar_cv.requests.post", return_value=mock_resp):
        from descargar_cv import get_download_url
        with pytest.raises(Exception, match="401"):
            get_download_url("bad-key")


def test_download_file_sends_range_header_on_resume(tmp_path: Path):
    part_file = tmp_path / "file.part"
    already_downloaded = b"first_chunk"
    part_file.write_bytes(already_downloaded)

    new_chunk = b"_second_chunk"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content.return_value = [new_chunk]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured = {}

    def fake_get(url, headers=None, stream=None):
        captured["headers"] = headers or {}
        return mock_resp

    with patch("descargar_cv.requests.get", side_effect=fake_get):
        from descargar_cv import download_file
        download_file("https://storage.example.com/file", part_file, 100)

    assert captured["headers"].get("Range") == f"bytes={len(already_downloaded)}-"
    assert part_file.read_bytes() == already_downloaded + new_chunk


def test_download_file_no_range_header_on_fresh_start(tmp_path: Path):
    part_file = tmp_path / "file.part"

    chunk = b"full_content"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content.return_value = [chunk]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured = {}

    def fake_get(url, headers=None, stream=None):
        captured["headers"] = headers or {}
        return mock_resp

    with patch("descargar_cv.requests.get", side_effect=fake_get):
        from descargar_cv import download_file
        download_file("https://storage.example.com/file", part_file, 100)

    assert "Range" not in captured["headers"]
    assert part_file.read_bytes() == chunk


# --- Integration test (requires MDC_API_KEY in .env + internet) ---

@pytest.mark.integration
def test_get_download_url_integration():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from descargar_cv import get_download_url
    result = get_download_url(os.environ["MDC_API_KEY"])
    assert "downloadUrl" in result
    assert result["sizeBytes"] > 0
```

- [ ] **Step 4: Run tests to confirm they fail**

```
uv run pytest tests/test_descargar_cv.py -v
```
Expected: `ImportError` — `descargar_cv` does not exist yet.

- [ ] **Step 5: Create `descargar_cv.py`**

```python
import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Verify this URL with the diagnostic in Step 2 and update if needed
MDC_API_BASE = "https://mozilladatacollective.com/api/v1"
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


def download_file(url: str, dest: Path, total_size: int) -> None:
    offset = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
    mode = "ab" if offset > 0 else "wb"

    with requests.get(url, headers=headers, stream=True) as resp:
        resp.raise_for_status()
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


def verify_checksum(path: Path, expected: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def extract_archive(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga Common Voice Scripted Speech 26.0 Spanish a ./data/cv-raw/"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-descarga aunque ya exista ./data/cv-raw/"
    )
    args = parser.parse_args()

    if OUTPUT_DIR.exists() and not args.force:
        print(f"El corpus ya existe en {OUTPUT_DIR}. Usa --force para re-descargar.")
        return

    api_key = os.environ.get("MDC_API_KEY", "")
    if not api_key:
        print("Error: MDC_API_KEY no encontrado en .env")
        sys.exit(1)

    print("Obteniendo URL de descarga...")
    info = get_download_url(api_key)
    url: str = info["downloadUrl"]
    total_size: int = info["sizeBytes"]
    checksum: str = info["checksum"]

    PART_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PART_FILE.exists():
        offset = PART_FILE.stat().st_size
        print(f"Reanudando desde {offset / 1e9:.2f} GB de {total_size / 1e9:.2f} GB...")
    else:
        print(f"Descargando {total_size / 1e9:.2f} GB...")

    download_file(url, PART_FILE, total_size)

    print("Verificando checksum SHA-256...")
    if not verify_checksum(PART_FILE, checksum):
        PART_FILE.unlink()
        print("Error: checksum no coincide. Archivo eliminado. Vuelve a ejecutar.")
        sys.exit(1)

    print(f"Extrayendo a {OUTPUT_DIR}...")
    extract_archive(PART_FILE, OUTPUT_DIR)
    PART_FILE.unlink()
    print(f"Corpus disponible en {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to confirm they pass**

```
uv run pytest tests/test_descargar_cv.py -v
```
Expected: 7 unit tests PASS, 1 integration test SKIPPED.

- [ ] **Step 7: Verify linting and types**

```
uv run ruff check descargar_cv.py
uv run mypy descargar_cv.py
```
Expected: no errors.

---

### Task 2: `cv_metadata.py` — rewrite for local TSV

**Files:**
- Modify: `cv_metadata.py` (full rewrite)
- Modify: `tests/test_cv_metadata.py` (full rewrite)

**Interfaces:**
- Produces:
  - `find_clips_tsv(data_dir: Path) -> Path` — finds first TSV with required columns via `rglob`
  - `get_top_female_speakers(data_dir: Path, n: int = 20) -> pd.DataFrame` — columns: `client_id` (str), `clips` (int), `minutes` (float)

- [ ] **Step 1: Rewrite `tests/test_cv_metadata.py`**

Replace the entire file with:

```python
from pathlib import Path

import pytest

TSV_CONTENT = (
    "client_id\tpath\tgender\tsentence\n"
    "abc123\tclips/001.mp3\tfemale\tHola mundo.\n"
    "abc123\tclips/002.mp3\tfemale\tEl cielo es azul.\n"
    "def456\tclips/003.mp3\tmale\tEl río corre.\n"
    "xyz789\tclips/004.mp3\tfemale\tLa casa.\n"
)


def test_find_clips_tsv_finds_file_in_root(tmp_path: Path):
    tsv = tmp_path / "clips.tsv"
    tsv.write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import find_clips_tsv
    assert find_clips_tsv(tmp_path) == tsv


def test_find_clips_tsv_finds_file_in_subdirectory(tmp_path: Path):
    subdir = tmp_path / "corpus" / "es"
    subdir.mkdir(parents=True)
    tsv = subdir / "validated.tsv"
    tsv.write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import find_clips_tsv
    assert find_clips_tsv(tmp_path) == tsv


def test_find_clips_tsv_raises_when_not_found(tmp_path: Path):
    from cv_metadata import find_clips_tsv
    with pytest.raises(FileNotFoundError, match="descargar-cv"):
        find_clips_tsv(tmp_path)


def test_get_top_female_speakers_columns(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers
    df = get_top_female_speakers(tmp_path, n=10)
    assert list(df.columns) == ["client_id", "clips", "minutes"]


def test_get_top_female_speakers_excludes_males(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers
    df = get_top_female_speakers(tmp_path)
    assert "def456" not in df["client_id"].values


def test_get_top_female_speakers_sorted_desc(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers
    df = get_top_female_speakers(tmp_path)
    assert df.iloc[0]["client_id"] == "abc123"
    clips = df["clips"].tolist()
    assert clips == sorted(clips, reverse=True)


def test_get_top_female_speakers_minutes(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers
    df = get_top_female_speakers(tmp_path)
    # abc123 has 2 clips × 5.0s / 60 = 0.1667 min
    assert df.iloc[0]["clips"] == 2
    assert df.iloc[0]["minutes"] == pytest.approx(2 * 5.0 / 60)


def test_get_top_female_speakers_respects_n(tmp_path: Path):
    lines = ["client_id\tpath\tgender\tsentence"]
    for i in range(10):
        lines.append(f"speaker_{i:03d}\tclips/{i:03d}.mp3\tfemale\tText {i}")
    (tmp_path / "clips.tsv").write_text("\n".join(lines), encoding="utf-8")
    from cv_metadata import get_top_female_speakers
    df = get_top_female_speakers(tmp_path, n=3)
    assert len(df) == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_cv_metadata.py -v
```
Expected: tests fail — the current `cv_metadata.py` has the old HuggingFace signature `get_top_female_speakers(n)`, not `get_top_female_speakers(data_dir, n)`.

- [ ] **Step 3: Rewrite `cv_metadata.py`**

Replace the entire file with:

```python
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
        f"No se encontró un TSV de clips de Common Voice en {data_dir}. "
        "Ejecuta: task descargar-cv"
    )


def get_top_female_speakers(data_dir: Path, n: int = 20) -> pd.DataFrame:
    """Return top-n female speakers sorted by estimated minutes."""
    tsv_path = find_clips_tsv(data_dir)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, usecols=["client_id", "gender"])
    females = df[df["gender"] == "female"]
    counts = (
        females.groupby("client_id", as_index=False)
        .size()
        .rename(columns={"size": "clips"})
    )
    counts["clips"] = counts["clips"].astype(int)
    counts["minutes"] = counts["clips"] * SECONDS_PER_CLIP / 60
    return counts.sort_values("minutes", ascending=False).head(n).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
uv run pytest tests/test_cv_metadata.py -v
```
Expected: 8 unit tests PASS.

- [ ] **Step 5: Verify linting and types**

```
uv run ruff check cv_metadata.py
uv run mypy cv_metadata.py
```
Expected: no errors.

---

### Task 3: `analizar_cv.py` — add data-dir guard

**Files:**
- Modify: `analizar_cv.py`

**Interfaces:**
- Consumes: `get_top_female_speakers(data_dir: Path, n: int = 20) -> pd.DataFrame` from `cv_metadata` (Task 2)

- [ ] **Step 1: Replace `analizar_cv.py`**

Replace the entire file with:

```python
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
print(f"{'#':<4} {'client_id':<20} {'clips':>8} {'minutos':>10}")
print("-" * 46)
for i, row in df.iterrows():
    short_id = row["client_id"][:16] + "..." if len(row["client_id"]) > 16 else row["client_id"]
    print(f"{i + 1:<4} {short_id:<20} {int(row['clips']):>8} {row['minutes']:>10.1f}")

best = df.iloc[0]["client_id"]
print("\nPara exportar la hablante con más audio:")
print(f"  uv run exportar_cv.py --speaker {best}")
```

- [ ] **Step 2: Verify linting**

```
uv run ruff check analizar_cv.py
```
Expected: no errors.

---

### Task 4: `exportar_cv.py` — rewrite for local MP3 files

**Files:**
- Modify: `exportar_cv.py` (full rewrite)
- Modify: `tests/test_exportar_cv.py` (full rewrite)

**Interfaces:**
- Consumes:
  - `find_clips_tsv(data_dir: Path) -> Path` from `cv_metadata`
  - `decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]` from `exportar_dataset`
  - `write_metadata_csv(entries: list[tuple[str, str]], path: Path) -> None` from `exportar_dataset`
  - `process_clip(audio: np.ndarray, source_sr: int) -> tuple[np.ndarray, int]` from `audio_utils`
  - `save_wav(audio: np.ndarray, sr: int, path: Path) -> None` from `audio_utils`
- Produces:
  - `build_wav_filename(client_id: str, index: int) -> str` — local function, returns `f"{client_id[:8]}_{index:04d}.wav"`
  - `export_speaker(client_id: str, data_dir: Path, output_dir: Path) -> None`

- [ ] **Step 1: Rewrite `tests/test_exportar_cv.py`**

Replace the entire file with:

```python
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _make_wav_bytes(duration_s: float = 0.5, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="wav", subtype="FLOAT")
    return buf.getvalue()


TSV_CONTENT = (
    "client_id\tpath\tgender\tsentence\n"
    "abc12345xyz\t001.wav\tfemale\tHola mundo.\n"
    "abc12345xyz\t002.wav\tfemale\tEl cielo es azul.\n"
    "other_speaker\t003.wav\tfemale\tOtra hablante.\n"
)


def _make_corpus(tmp_path: Path) -> Path:
    """Create a minimal fake corpus directory with TSV + WAV clips."""
    corpus = tmp_path / "cv-raw"
    corpus.mkdir()
    clips = corpus / "clips"
    clips.mkdir()
    (corpus / "validated.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    (clips / "001.wav").write_bytes(_make_wav_bytes())
    (clips / "002.wav").write_bytes(_make_wav_bytes())
    (clips / "003.wav").write_bytes(_make_wav_bytes())
    return corpus


def test_build_wav_filename_uses_first_8_chars():
    from exportar_cv import build_wav_filename
    assert build_wav_filename("abc12345xyz", 1) == "abc12345_0001.wav"


def test_build_wav_filename_zero_padded():
    from exportar_cv import build_wav_filename
    assert build_wav_filename("abcdefghijk", 42) == "abcdefgh_0042.wav"


def test_export_speaker_produces_wavs_and_metadata(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    output = tmp_path / "output"
    from exportar_cv import export_speaker
    export_speaker("abc12345xyz", corpus, output)

    metadata = (output / "metadata.csv").read_text(encoding="utf-8")
    lines = metadata.strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == "wavs/abc12345_0001.wav|Hola mundo."
    assert lines[1] == "wavs/abc12345_0002.wav|El cielo es azul."


def test_export_speaker_wav_files_exist(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    output = tmp_path / "output"
    from exportar_cv import export_speaker
    export_speaker("abc12345xyz", corpus, output)

    assert (output / "wavs" / "abc12345_0001.wav").exists()
    assert (output / "wavs" / "abc12345_0002.wav").exists()


def test_export_speaker_only_exports_requested_speaker(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    output = tmp_path / "output"
    from exportar_cv import export_speaker
    export_speaker("abc12345xyz", corpus, output)

    wavs = list((output / "wavs").glob("*.wav"))
    assert len(wavs) == 2  # only 2 clips, not 3 (other_speaker excluded)


def test_export_speaker_exits_when_speaker_not_found(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    output = tmp_path / "output"
    from exportar_cv import export_speaker
    with pytest.raises(SystemExit):
        export_speaker("nonexistent_speaker", corpus, output)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/test_exportar_cv.py -v
```
Expected: tests fail — current `exportar_cv.py` has wrong signature and HuggingFace imports.

- [ ] **Step 3: Rewrite `exportar_cv.py`**

Replace the entire file with:

```python
import argparse
import sys
from pathlib import Path

import numpy as np
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
        "--data-dir", default="./data/cv-raw",
        help="Directorio del corpus descargado (default: ./data/cv-raw)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Directorio de salida (default: ./data/<client_id[:8]>)"
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
uv run pytest tests/test_exportar_cv.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Run the full test suite to check no regressions**

```
uv run pytest tests/ -v
```
Expected: all unit tests PASS (previously passing MLS tests still pass).

- [ ] **Step 6: Verify linting and types**

```
uv run ruff check exportar_cv.py
uv run mypy exportar_cv.py
```
Expected: no errors.

---

### Task 5: `LICENSES.md` + `Taskfile.yml` + `.env` updates

**Files:**
- Modify: `LICENSES.md`
- Modify: `Taskfile.yml`
- Document: `.env` (add `MDC_API_KEY=` entry — do NOT commit this file)

- [ ] **Step 1: Update `LICENSES.md`**

Replace the Mozilla Common Voice section with:

```markdown
## Mozilla Common Voice

- **Fuente:** Mozilla Data Collective — Common Voice Scripted Speech 26.0 - Spanish
- **Dataset ID:** `cmqim2spa00synr071fcp7av0`
- **Licencia:** [CC0 1.0 Universal (dominio público)](https://creativecommons.org/publicdomain/zero/1.0/)
- No requiere atribución. Puedes usar, modificar y distribuir el dataset sin restricciones.
- **Nota:** Los datos ya no están disponibles en HuggingFace desde octubre 2025. Se descargan desde [Mozilla Data Collective](https://mozilladatacollective.com).
```

- [ ] **Step 2: Update `Taskfile.yml` — add `descargar-cv` task**

Add after the `sync` task:

```yaml
  descargar-cv:
    desc: "Descarga Common Voice Scripted Speech 26.0 Spanish a ./data/cv-raw/ (48 GB, reanudable)"
    cmds:
      - uv run descargar_cv.py
```

- [ ] **Step 3: Document `MDC_API_KEY` in `.env`**

Add to `.env` (the file is gitignored):
```
MDC_API_KEY=your_api_key_here
```

- [ ] **Step 4: Verify Taskfile syntax**

```
task --list
```
Expected: `descargar-cv`, `analizar-cv`, `exportar-cv` all appear in the list.

- [ ] **Step 5: Final full test suite run**

```
uv run pytest tests/ -v
```
Expected: all unit tests PASS with no regressions across MLS and CV tests.

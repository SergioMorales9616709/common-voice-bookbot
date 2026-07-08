# Common Voice Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mozilla Common Voice as a second audio source for LJSpeech export, parallel to MLS, without modifying any existing files.

**Architecture:** Three new top-level scripts mirror the MLS pattern — `cv_metadata.py` handles streaming aggregation of speakers, `analizar_cv.py` prints the ranked speaker list, and `exportar_cv.py` exports a single speaker's clips to LJSpeech format. All audio processing goes through the existing `audio_utils.py`.

**Tech Stack:** `datasets>=3.6.0` (HuggingFace streaming), `pandas>=3.0.3`, `tqdm`, `python-dotenv`, `audio_utils.process_clip` / `save_wav`

## Global Constraints

- Python 3.13+
- Do NOT modify: `audio_utils.py`, `mls_metadata.py`, `analizar_mls.py`, `exportar_dataset.py`
- Output format: LJSpeech `wavs/<prefix>_NNNN.wav` + `metadata.csv` with `filename|text` rows, no header
- CV dataset: `mozilla-foundation/common_voice_17_0`, language subset `es`
- Audio target: 22050 Hz mono PCM_16, EBU R128 -23 LUFS (via `audio_utils.process_clip`)
- Unit tests run without network access; integration tests use `@pytest.mark.integration`
- Run tests: `uv run pytest tests/ -v` (unit) | `uv run pytest tests/ -m integration -v` (integration)

---

### Task 1: `cv_metadata.py` — female speaker aggregation

**Files:**
- Create: `cv_metadata.py`
- Create: `tests/test_cv_metadata.py`

**Interfaces:**
- Produces:
  - `get_top_female_speakers(n: int = 20) -> pd.DataFrame` — columns: `client_id` (str), `clips` (int), `minutes` (float)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cv_metadata.py`:

```python
import pytest
from unittest.mock import patch

import pandas as pd


def test_get_top_female_speakers_columns():
    with patch("cv_metadata._iter_female_rows", return_value=iter(["abc123", "abc123", "def456"])):
        from cv_metadata import get_top_female_speakers
        df = get_top_female_speakers(n=10)

    assert list(df.columns) == ["client_id", "clips", "minutes"]


def test_get_top_female_speakers_sorted_desc():
    # abc123 appears 3x → should be first
    with patch("cv_metadata._iter_female_rows", return_value=iter(["abc123"] * 3 + ["def456"])):
        from cv_metadata import get_top_female_speakers
        df = get_top_female_speakers()

    assert df.iloc[0]["client_id"] == "abc123"
    clips = df["clips"].tolist()
    assert clips == sorted(clips, reverse=True)


def test_get_top_female_speakers_respects_n():
    ids = [f"speaker_{i}" for i in range(10)]
    with patch("cv_metadata._iter_female_rows", return_value=iter(ids)):
        from cv_metadata import get_top_female_speakers
        df = get_top_female_speakers(n=3)

    assert len(df) == 3


def test_get_top_female_speakers_minutes_derived_from_clips():
    # 12 clips × 5.0s / 60 = 1.0 minute
    with patch("cv_metadata._iter_female_rows", return_value=iter(["abc123"] * 12)):
        from cv_metadata import get_top_female_speakers
        df = get_top_female_speakers()

    assert df.iloc[0]["clips"] == 12
    assert df.iloc[0]["minutes"] == pytest.approx(1.0)


def test_get_top_female_speakers_empty_returns_empty_df():
    with patch("cv_metadata._iter_female_rows", return_value=iter([])):
        from cv_metadata import get_top_female_speakers
        df = get_top_female_speakers()

    assert df.empty
    assert list(df.columns) == ["client_id", "clips", "minutes"]


# --- Integration test (requires HF_TOKEN + internet) ---

@pytest.mark.integration
def test_get_top_female_speakers_integration():
    from cv_metadata import get_top_female_speakers
    df = get_top_female_speakers(n=5)
    assert not df.empty
    assert list(df.columns) == ["client_id", "clips", "minutes"]
    assert df["minutes"].iloc[0] > 0
    assert pd.api.types.is_string_dtype(df["client_id"])
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_cv_metadata.py -v
```

Expected: `ImportError` — `cv_metadata` does not exist yet.

- [ ] **Step 3: Create `cv_metadata.py`**

```python
from collections import defaultdict
from collections.abc import Iterator

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

CV_DATASET = "mozilla-foundation/common_voice_17_0"
CV_LANGUAGE = "es"
CV_SPLITS = ["train", "validation", "test"]
SECONDS_PER_CLIP = 5.0


def _iter_female_rows() -> Iterator[str]:
    """Stream CV Spanish metadata, yield client_id for each female-gendered row."""
    for split in CV_SPLITS:
        ds = load_dataset(CV_DATASET, CV_LANGUAGE, split=split, streaming=True, trust_remote_code=True)
        ds = ds.remove_columns(["audio"])
        for row in ds:
            if row.get("gender") == "female":
                yield row["client_id"]


def get_top_female_speakers(n: int = 20) -> pd.DataFrame:
    """Return top-n female speakers sorted by estimated minutes of audio."""
    counts: dict[str, int] = defaultdict(int)
    for client_id in _iter_female_rows():
        counts[client_id] += 1

    if not counts:
        return pd.DataFrame(columns=["client_id", "clips", "minutes"])

    df = pd.DataFrame(
        [
            {"client_id": cid, "clips": cnt, "minutes": cnt * SECONDS_PER_CLIP / 60}
            for cid, cnt in counts.items()
        ]
    )
    return df.sort_values("minutes", ascending=False).head(n).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_cv_metadata.py -v
```

Expected: 5 unit tests PASS, integration test skipped (no `-m integration` flag).

---

### Task 2: `analizar_cv.py` — top speaker CLI

**Files:**
- Create: `analizar_cv.py`

**Interfaces:**
- Consumes: `get_top_female_speakers(n: int = 20) -> pd.DataFrame` from `cv_metadata`

- [ ] **Step 1: Create `analizar_cv.py`**

```python
import sys

from cv_metadata import get_top_female_speakers

print("\n--- Descargando metadatos de hablantes Common Voice Spanish ---")
df = get_top_female_speakers(n=10)

if df.empty:
    print("Error: no se pudieron obtener datos de hablantes.")
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

- [ ] **Step 2: Verify linting passes**

```
uv run ruff check analizar_cv.py
```

Expected: no errors.

---

### Task 3: `exportar_cv.py` — LJSpeech export

**Files:**
- Create: `exportar_cv.py`
- Create: `tests/test_exportar_cv.py`

**Interfaces:**
- Consumes: `process_clip(audio: np.ndarray, source_sr: int) -> tuple[np.ndarray, int]` and `save_wav(audio: np.ndarray, sr: int, path: Path)` from `audio_utils`
- Produces: `<output_dir>/wavs/<prefix>_NNNN.wav` + `<output_dir>/metadata.csv`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exportar_cv.py`:

```python
from pathlib import Path

from exportar_cv import build_wav_filename, write_metadata_csv


def test_build_wav_filename_uses_first_8_chars():
    client_id = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    assert build_wav_filename(client_id, 1) == "a1b2c3d4_0001.wav"


def test_build_wav_filename_zero_padded():
    assert build_wav_filename("abcdefghijk", 42) == "abcdefgh_0042.wav"


def test_write_metadata_csv_no_header(tmp_path: Path):
    entries = [
        ("wavs/a1b2c3d4_0001.wav", "Hola mundo."),
        ("wavs/a1b2c3d4_0002.wav", "El cielo es azul."),
    ]
    out = tmp_path / "metadata.csv"
    write_metadata_csv(entries, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == "wavs/a1b2c3d4_0001.wav|Hola mundo."
    assert lines[1] == "wavs/a1b2c3d4_0002.wav|El cielo es azul."


def test_write_metadata_csv_pipe_separator(tmp_path: Path):
    entries = [("wavs/001.wav", "Hola")]
    out = tmp_path / "metadata.csv"
    write_metadata_csv(entries, out)
    content = out.read_text(encoding="utf-8")
    assert "|" in content
    assert "," not in content


def test_write_metadata_csv_utf8_encoding(tmp_path: Path):
    entries = [("wavs/001.wav", "El niño jugó en el jardín.")]
    out = tmp_path / "metadata.csv"
    write_metadata_csv(entries, out)
    assert "El niño jugó en el jardín." in out.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_exportar_cv.py -v
```

Expected: `ImportError` — `exportar_cv` does not exist yet.

- [ ] **Step 3: Create `exportar_cv.py`**

```python
import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from audio_utils import process_clip, save_wav

load_dotenv()

CV_DATASET = "mozilla-foundation/common_voice_17_0"
CV_LANGUAGE = "es"
CV_SPLITS = ["train", "validation", "test"]
MIN_DURATION_SECONDS = 0.5


def build_wav_filename(client_id: str, index: int) -> str:
    return f"{client_id[:8]}_{index:04d}.wav"


def write_metadata_csv(entries: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for filename, text in entries:
            f.write(f"{filename}|{text}\n")


def export_speaker(client_id: str, output_dir: Path) -> None:
    wavs_dir = output_dir / "wavs"
    all_entries: list[tuple[str, str]] = []
    clip_index = 1
    total_skipped = 0

    for split in CV_SPLITS:
        print(f"\n  Split '{split}'...")
        ds = load_dataset(CV_DATASET, CV_LANGUAGE, split=split, streaming=True, trust_remote_code=True)
        speaker_rows = (row for row in ds if row["client_id"] == client_id)

        split_count = 0
        for row in tqdm(speaker_rows, desc=f"    clips", unit="clip", leave=False):
            audio_data = row.get("audio")
            if not audio_data:
                total_skipped += 1
                continue

            audio_array: np.ndarray = audio_data["array"]
            source_sr: int = audio_data["sampling_rate"]

            if audio_array is None or len(audio_array) == 0:
                total_skipped += 1
                continue

            duration = len(audio_array) / source_sr
            if duration < MIN_DURATION_SECONDS:
                total_skipped += 1
                continue

            try:
                processed, target_sr = process_clip(audio_array.astype(np.float32), source_sr)
                wav_name = build_wav_filename(client_id, clip_index)
                save_wav(processed, target_sr, wavs_dir / wav_name)
            except Exception as e:
                print(f"      Warning: process error ({e})")
                total_skipped += 1
                continue

            sentence = str(row.get("sentence", "")).strip()
            all_entries.append((f"wavs/{wav_name}", sentence))
            clip_index += 1
            split_count += 1

        if split_count:
            print(f"      +{split_count} clips (total: {len(all_entries)})")

    if not all_entries:
        print(f"\nError: no se encontraron clips para client_id='{client_id}'.")
        print("Ejecuta analizar_cv.py para ver client_ids válidos.")
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
        "--output", default=None, help="Directorio de salida (default: ./data/<client_id[:8]>)"
    )
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path("./data") / args.speaker[:8]
    export_speaker(args.speaker, output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_exportar_cv.py -v
```

Expected: 5 unit tests PASS.

- [ ] **Step 5: Verify linting and types pass**

```
uv run ruff check cv_metadata.py analizar_cv.py exportar_cv.py
uv run mypy cv_metadata.py exportar_cv.py
```

Expected: no errors.

---

### Task 4: `LICENSES.md` — license documentation

**Files:**
- Create: `LICENSES.md`

- [ ] **Step 1: Create `LICENSES.md`**

```markdown
# Licencias de las fuentes de datos

## Multilingual LibriSpeech (MLS)

- **Fuente:** `facebook/multilingual_librispeech` en HuggingFace
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Atribución requerida:** Facebook AI Research / LibriVox
- El audio proviene de grabaciones de voluntarios de LibriVox y fue redistribuido bajo CC BY 4.0 por Facebook.

## Mozilla Common Voice

- **Fuente:** `mozilla-foundation/common_voice_17_0` en HuggingFace
- **Licencia:** [CC0 1.0 Universal (dominio público)](https://creativecommons.org/publicdomain/zero/1.0/)
- No requiere atribución. Puedes usar, modificar y distribuir el dataset sin restricciones.
```

- [ ] **Step 2: Verify the file is readable**

```
uv run python -c "from pathlib import Path; print(Path('LICENSES.md').read_text(encoding='utf-8'))"
```

Expected: full content printed without errors.

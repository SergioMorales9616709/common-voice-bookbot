# MLS Spanish Dataset Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two scripts — `analizar_mls.py` y `exportar_dataset.py` — que identifican la mejor hablante femenina en MLS Spanish y exportan sus clips como un dataset LJSpeech listo para entrenar piper-tts.

**Architecture:** `analizar_mls.py` descarga el `metainfo.txt` del repositorio MLS en HuggingFace y muestra un ranking de hablantes femeninas por minutos totales. `exportar_dataset.py` recibe un `speaker_id` por CLI, hace streaming del dataset de esa hablante, aplica resample + loudnorm a cada clip, y escribe WAVs + `metadata.csv` en formato LJSpeech.

**Tech Stack:** Python 3.13, uv, huggingface-hub, datasets (HuggingFace streaming), librosa, pyloudnorm, soundfile, pandas, pytest

## Global Constraints

- Python 3.13, gestionado con `uv` — siempre ejecutar como `uv run <script>`
- Target audio: WAV mono 22050 Hz PCM_16
- Target loudness: -23 LUFS (EBU R128)
- Duración mínima por clip: 0.5 segundos (descartar más cortos)
- `metadata.csv`: sin encabezado, separador `|`, formato `wavs/<filename>.wav|<texto>`
- Directorio de salida: `./data/dataset/` (relativo a la raíz del proyecto)
- Paths: usar `pathlib.Path` para compatibilidad Windows
- Tests: ejecutar con `uv run pytest`

---

### Task 1: Agregar dependencias y configurar proyecto

**Files:**
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces: `pyloudnorm`, `datasets`, `pytest` disponibles para importar

- [ ] **Step 1: Actualizar pyproject.toml**

Reemplazar el contenido completo de `pyproject.toml`:

```toml
[project]
name = "common-voice-bookbot"
version = "0.1.0"
description = "Tooling to extract and prepare MLS Spanish voice data for piper-tts training"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "huggingface-hub>=1.20.1",
    "librosa>=0.11.0",
    "pandas>=3.0.3",
    "soundfile>=0.14.0",
    "tqdm>=4.68.3",
    "pyarrow>=20.0.0",
    "pyloudnorm>=0.1.1",
    "datasets>=3.6.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 2: Instalar dependencias**

```bash
uv sync
```

Expected: instala `pyloudnorm`, `datasets`, `pytest` y sus dependencias.

- [ ] **Step 3: Verificar imports**

```bash
uv run python -c "import pyloudnorm; import datasets; import pytest; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Actualizar CLAUDE.md (sección Key scripts)**

Reemplazar la sección `## Key scripts` en `CLAUDE.md`:

```markdown
## Key scripts

- `analizar_mls.py` — descarga el metainfo de MLS Spanish desde HuggingFace y muestra el top 10 de hablantes femeninas por minutos totales. Ejecutar una vez para elegir el `speaker_id` a usar.
- `exportar_dataset.py` — exporta todos los clips de una hablante a `./data/dataset/` en formato LJSpeech (WAVs 22050 Hz + metadata.csv). Uso: `uv run exportar_dataset.py --speaker <speaker_id>`
- `analizar_voces.py` — script legacy para el sample de Bookbot Common Voice. Se mantiene como referencia.
- `main.py` — placeholder.
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock CLAUDE.md
git commit -m "feat: add pyloudnorm, datasets, pytest dependencies"
```

---

### Task 2: Módulo de metadatos de hablantes (`mls_metadata.py`)

**Files:**
- Create: `mls_metadata.py`
- Create: `tests/test_mls_metadata.py`

**Interfaces:**
- Produces:
  - `parse_metainfo(content: str) -> pd.DataFrame` — columnas: `speaker_id` (str), `gender` (str: "M"/"F"), `minutes` (float)
  - `download_speaker_metadata() -> pd.DataFrame` — descarga de HuggingFace y llama a `parse_metainfo`
  - `get_female_speakers(df: pd.DataFrame) -> pd.DataFrame` — filtra gender=="F", ordenado por minutes desc

- [ ] **Step 1: Escribir tests que fallen**

Crear `tests/test_mls_metadata.py`:

```python
import pandas as pd
import pytest
from mls_metadata import parse_metainfo, get_female_speakers

SAMPLE_METAINFO = (
    "2138\ttrain\t127.4\tSome Name\tF\n"
    "3421\ttrain\t98.2\tAnother Name\tM\n"
    "5491\ttrain\t84.1\tThird Reader\tF\n"
    "6012\tdev\t12.3\tFourth Reader\tF\n"
)


def test_parse_metainfo_columns():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert list(df.columns) == ["speaker_id", "gender", "minutes"]


def test_parse_metainfo_speaker_ids_are_strings():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert df["speaker_id"].dtype == object


def test_parse_metainfo_row_count():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert len(df) == 4


def test_parse_metainfo_values():
    df = parse_metainfo(SAMPLE_METAINFO)
    row = df[df["speaker_id"] == "2138"].iloc[0]
    assert row["gender"] == "F"
    assert row["minutes"] == pytest.approx(127.4)


def test_get_female_speakers_excludes_males():
    df = parse_metainfo(SAMPLE_METAINFO)
    females = get_female_speakers(df)
    assert all(females["gender"] == "F")
    assert "3421" not in females["speaker_id"].values


def test_get_female_speakers_sorted_desc():
    df = parse_metainfo(SAMPLE_METAINFO)
    females = get_female_speakers(df)
    minutes = females["minutes"].tolist()
    assert minutes == sorted(minutes, reverse=True)
```

- [ ] **Step 2: Confirmar que los tests fallan**

```bash
uv run pytest tests/test_mls_metadata.py -v
```

Expected: `ModuleNotFoundError: No module named 'mls_metadata'`

- [ ] **Step 3: Implementar mls_metadata.py**

Crear `mls_metadata.py`:

```python
import io
import pandas as pd
from huggingface_hub import hf_hub_download


def parse_metainfo(content: str) -> pd.DataFrame:
    df = pd.read_csv(
        io.StringIO(content),
        sep="\t",
        header=None,
        names=["speaker_id", "subset", "minutes", "name", "gender"],
        dtype={"speaker_id": str},
    )
    return df[["speaker_id", "gender", "minutes"]]


def get_female_speakers(df: pd.DataFrame) -> pd.DataFrame:
    females = df[df["gender"] == "F"].copy()
    return females.sort_values("minutes", ascending=False).reset_index(drop=True)


def download_speaker_metadata() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="facebook/multilingual_librispeech",
        filename="spanish/metainfo.txt",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return parse_metainfo(content)
```

- [ ] **Step 4: Confirmar que los tests pasan**

```bash
uv run pytest tests/test_mls_metadata.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mls_metadata.py tests/test_mls_metadata.py
git commit -m "feat: add MLS speaker metadata module"
```

---

### Task 3: Utilidades de procesamiento de audio (`audio_utils.py`)

**Files:**
- Create: `audio_utils.py`
- Create: `tests/test_audio_utils.py`

**Interfaces:**
- Produces:
  - `process_clip(audio: np.ndarray, source_sr: int) -> tuple[np.ndarray, int]` — retorna (audio resampleado + normalizado, 22050)
  - `save_wav(audio: np.ndarray, sr: int, path: Path) -> None` — escribe WAV PCM_16

- [ ] **Step 1: Escribir tests que fallen**

Crear `tests/test_audio_utils.py`:

```python
import numpy as np
import pytest
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path
from audio_utils import process_clip, save_wav

TARGET_SR = 22050


def make_sine(seconds: float = 1.5, sr: int = 16000, amplitude: float = 0.1) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def test_process_clip_resamples_to_22050():
    audio = make_sine(sr=16000)
    result, sr = process_clip(audio, 16000)
    assert sr == TARGET_SR
    assert result.shape[0] == pytest.approx(TARGET_SR * 1.5, abs=10)


def test_process_clip_normalizes_loudness():
    audio = make_sine(seconds=1.5, sr=16000, amplitude=0.001)
    result, sr = process_clip(audio, 16000)
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(result)
    assert abs(measured - (-23.0)) < 1.5


def test_process_clip_passthrough_when_sr_matches():
    audio = make_sine(sr=TARGET_SR)
    result, sr = process_clip(audio, TARGET_SR)
    assert sr == TARGET_SR
    assert len(result) == len(audio)


def test_save_wav_creates_file(tmp_path: Path):
    audio = make_sine(sr=TARGET_SR)
    out = tmp_path / "test.wav"
    save_wav(audio, TARGET_SR, out)
    assert out.exists()


def test_save_wav_correct_samplerate(tmp_path: Path):
    audio = make_sine(sr=TARGET_SR)
    out = tmp_path / "test.wav"
    save_wav(audio, TARGET_SR, out)
    info = sf.info(str(out))
    assert info.samplerate == TARGET_SR


def test_save_wav_pcm16_subtype(tmp_path: Path):
    audio = make_sine(sr=TARGET_SR)
    out = tmp_path / "test.wav"
    save_wav(audio, TARGET_SR, out)
    info = sf.info(str(out))
    assert "PCM_16" in info.subtype
```

- [ ] **Step 2: Confirmar que los tests fallan**

```bash
uv run pytest tests/test_audio_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_utils'`

- [ ] **Step 3: Implementar audio_utils.py**

Crear `audio_utils.py`:

```python
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path

TARGET_SR = 22050
TARGET_LUFS = -23.0


def process_clip(audio: np.ndarray, source_sr: int) -> tuple[np.ndarray, int]:
    if source_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=TARGET_SR)
    meter = pyln.Meter(TARGET_SR)
    loudness = meter.integrated_loudness(audio)
    if np.isfinite(loudness):
        audio = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
    return audio, TARGET_SR


def save_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
```

- [ ] **Step 4: Confirmar que los tests pasan**

```bash
uv run pytest tests/test_audio_utils.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add audio_utils.py tests/test_audio_utils.py
git commit -m "feat: add audio processing utilities (resample + loudnorm)"
```

---

### Task 4: Script de análisis (`analizar_mls.py`)

**Files:**
- Create: `analizar_mls.py`

**Interfaces:**
- Consumes: `mls_metadata.download_speaker_metadata()`, `mls_metadata.get_female_speakers()`
- Produces: ranking en consola + comando sugerido

- [ ] **Step 1: Crear analizar_mls.py**

```python
from mls_metadata import download_speaker_metadata, get_female_speakers

print("\n--- Descargando metadatos de hablantes MLS Spanish ---")
df = download_speaker_metadata()

females = get_female_speakers(df)
top = females.head(10).reset_index(drop=True)

print("\nTop hablantes femeninas en MLS Spanish (por minutos de audio):\n")
print(f"{'#':<4} {'speaker_id':<14} {'minutos':>10}")
print("-" * 32)
for i, row in top.iterrows():
    print(f"{i + 1:<4} {row['speaker_id']:<14} {row['minutes']:>10.1f}")

best = top.iloc[0]["speaker_id"]
print(f"\nPara exportar la hablante con más audio:")
print(f"  uv run exportar_dataset.py --speaker {best}")
```

- [ ] **Step 2: Ejecutar manualmente**

```bash
uv run analizar_mls.py
```

Expected: descarga `metainfo.txt`, imprime tabla con ≥1 hablante femenina y el comando sugerido.

**Si falla con `EntryNotFoundError`**, verificar el path real del archivo en HuggingFace:

```bash
uv run python -c "
from huggingface_hub import list_repo_files
files = [f for f in list_repo_files('facebook/multilingual_librispeech', repo_type='dataset') if 'metainfo' in f.lower()]
print(files)
"
```

Actualizar el parámetro `filename` en `mls_metadata.download_speaker_metadata()` con el path correcto que devuelva ese comando.

- [ ] **Step 3: Commit**

```bash
git add analizar_mls.py
git commit -m "feat: add analizar_mls.py speaker ranking script"
```

---

### Task 5: Script de exportación (`exportar_dataset.py`)

**Files:**
- Create: `exportar_dataset.py`
- Create: `tests/test_exportar_dataset.py`

**Interfaces:**
- Consumes: `audio_utils.process_clip()`, `audio_utils.save_wav()`
- Consumes: CLI args `--speaker <speaker_id>`, `--output <path>` (default: `./data/dataset`)
- Produces: `<output>/wavs/<speaker_id>_<NNNN>.wav` + `<output>/metadata.csv`

- [ ] **Step 1: Escribir tests que fallen**

Crear `tests/test_exportar_dataset.py`:

```python
from pathlib import Path
from exportar_dataset import build_wav_filename, write_metadata_csv


def test_build_wav_filename_zero_padded():
    assert build_wav_filename("2138", 1) == "2138_0001.wav"
    assert build_wav_filename("2138", 42) == "2138_0042.wav"
    assert build_wav_filename("2138", 9999) == "2138_9999.wav"


def test_write_metadata_csv_no_header(tmp_path: Path):
    entries = [
        ("wavs/2138_0001.wav", "La casa está al lado del río."),
        ("wavs/2138_0002.wav", "El tren llega a las ocho."),
    ]
    out = tmp_path / "metadata.csv"
    write_metadata_csv(entries, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == "wavs/2138_0001.wav|La casa está al lado del río."
    assert lines[1] == "wavs/2138_0002.wav|El tren llega a las ocho."


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
    content = out.read_text(encoding="utf-8")
    assert "El niño jugó en el jardín." in content
```

- [ ] **Step 2: Confirmar que los tests fallan**

```bash
uv run pytest tests/test_exportar_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'exportar_dataset'`

- [ ] **Step 3: Implementar exportar_dataset.py**

Crear `exportar_dataset.py`:

```python
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
```

- [ ] **Step 4: Confirmar que los tests pasan**

```bash
uv run pytest tests/test_exportar_dataset.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Ejecutar todos los tests**

```bash
uv run pytest tests/ -v
```

Expected: 16 tests PASS (6 mls_metadata + 6 audio_utils + 4 exportar_dataset)

- [ ] **Step 6: Commit**

```bash
git add exportar_dataset.py tests/test_exportar_dataset.py
git commit -m "feat: add exportar_dataset.py LJSpeech export script"
```

---

## Self-Review

**Cobertura del spec:**

| Requisito del spec | Task |
|---|---|
| Descargar metainfo.txt de HuggingFace | Task 2 |
| Filtrar hablantes femeninas, ranking por minutos | Task 2 + Task 4 |
| Imprimir top 10 + comando sugerido | Task 4 |
| CLI `--speaker` argument | Task 5 |
| Resample a 22050 Hz mono | Task 3 |
| Loudness normalization -23 LUFS EBU R128 | Task 3 |
| WAV 16-bit PCM | Task 3 |
| metadata.csv sin encabezado, separador `|` | Task 5 |
| Omitir clips < 0.5 s | Task 5 |
| Error claro si speaker_id no existe | Task 5 |
| Agregar pyloudnorm + datasets | Task 1 |
| Actualizar CLAUDE.md | Task 1 |

**Consistencia de nombres entre tasks:**
- `process_clip` — definida Task 3, usada Task 5 ✓
- `save_wav` — definida Task 3, usada Task 5 ✓
- `write_metadata_csv` — definida Task 5, testeada Task 5 ✓
- `build_wav_filename` — definida Task 5, testeada Task 5 ✓
- `download_speaker_metadata` — definida Task 2, usada Task 4 ✓
- `get_female_speakers` — definida Task 2, usada Task 4 ✓

Sin gaps, sin placeholders, sin inconsistencias de tipos.

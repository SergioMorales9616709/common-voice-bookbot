# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Tooling to download, prepare, and export single-speaker female Spanish voice datasets from [Multilingual LibriSpeech (MLS)](https://huggingface.co/datasets/facebook/multilingual_librispeech) and [Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) on HuggingFace. The goal is to produce LJSpeech-format datasets ready for training a piper-tts ONNX voice model.

## Environment setup

This project uses `uv` for dependency management (Python 3.13).

```bash
uv sync          # install dependencies into .venv
uv run <script>  # run a script within the virtual environment
```

A `HF_TOKEN` env var is required for authenticated HuggingFace downloads. It is stored in `.env` (gitignored).

## Key scripts

### Multilingual LibriSpeech (MLS)

- `analizar_mls.py` — descarga el metainfo de MLS Spanish desde HuggingFace y muestra el top 10 de hablantes femeninas por minutos totales. Ejecutar una vez para elegir el `speaker_id` a usar.
- `exportar_dataset.py` — exporta todos los clips de una hablante a `./data/<speaker_id>/` en formato LJSpeech (WAVs 22050 Hz + metadata.csv). Uso: `uv run exportar_dataset.py --speaker <speaker_id>`

### Mozilla Common Voice

- `analizar_cv.py` — descarga metadatos de Common Voice Spanish y muestra el top 10 de hablantes femeninas por minutos estimados. Ejecutar una vez para elegir el `client_id` a usar.
- `exportar_cv.py` — exporta todos los clips de una hablante a `./data/<client_id[:8]>/` en formato LJSpeech (WAVs 22050 Hz + metadata.csv). Uso: `uv run exportar_cv.py --speaker <client_id>`

### Otros

- `analizar_voces.py` — script legacy para el sample de Bookbot Common Voice. Se mantiene como referencia.
- `main.py` — placeholder.

## Architecture notes

`analizar_mls.py` descarga el archivo `spanish/metainfo.txt` del repositorio MLS en HuggingFace Hub para obtener el ranking de hablantes sin descargar el audio (~220 GB). `exportar_dataset.py` usa la librería `datasets` con `streaming=True` para descargar solo los clips de la hablante elegida.

El audio se procesa en Python puro (sin ffmpeg): `librosa` para resample a 22050 Hz mono, `pyloudnorm` para normalización EBU R128 (-23 LUFS), y `soundfile` para escribir WAV PCM_16.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

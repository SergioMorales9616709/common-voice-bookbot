# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Tooling to download, inspect, and extract speaker data from the [Bookbot Common Voice 16.1 Spanish sample dataset](https://huggingface.co/datasets/bookbot/common_voice_16_1_es_sample) on HuggingFace. The goal is to identify individual speakers with enough clips for voice cloning or TTS training.

## Environment setup

This project uses `uv` for dependency management (Python 3.13).

```bash
uv sync          # install dependencies into .venv
uv run <script>  # run a script within the virtual environment
```

A `HF_TOKEN` env var is required for authenticated HuggingFace downloads. It is stored in `.env` (gitignored).

## Key scripts

- `analizar_mls.py` — descarga el metainfo de MLS Spanish desde HuggingFace y muestra el top 10 de hablantes femeninas por minutos totales. Ejecutar una vez para elegir el `speaker_id` a usar.
- `exportar_dataset.py` — exporta todos los clips de una hablante a `./data/dataset/` en formato LJSpeech (WAVs 22050 Hz + metadata.csv). Uso: `uv run exportar_dataset.py --speaker <speaker_id>`
- `analizar_voces.py` — script legacy para el sample de Bookbot Common Voice. Se mantiene como referencia.
- `main.py` — placeholder.

## Architecture notes

The dataset is downloaded via `huggingface_hub.snapshot_download` and cached locally by HuggingFace's cache mechanism. The metadata is in `metadata.csv` inside the snapshot directory. Speaker identity is tracked by the `client_id` column. Audio files are referenced in the metadata and can be loaded with `librosa` or `soundfile`.

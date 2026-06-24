# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Tooling to download, prepare, and export a single-speaker female Spanish voice dataset from [Multilingual LibriSpeech (MLS)](https://huggingface.co/datasets/facebook/multilingual_librispeech) on HuggingFace. The goal is to produce a LJSpeech-format dataset ready for training a piper-tts ONNX voice model.

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

`analizar_mls.py` descarga el archivo `spanish/metainfo.txt` del repositorio MLS en HuggingFace Hub para obtener el ranking de hablantes sin descargar el audio (~220 GB). `exportar_dataset.py` usa la librería `datasets` con `streaming=True` para descargar solo los clips de la hablante elegida.

El audio se procesa en Python puro (sin ffmpeg): `librosa` para resample a 22050 Hz mono, `pyloudnorm` para normalización EBU R128 (-23 LUFS), y `soundfile` para escribir WAV PCM_16.

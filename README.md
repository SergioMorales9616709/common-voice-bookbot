# common-voice-bookbot

Tooling to download and prepare a female Spanish voice dataset from Multilingual LibriSpeech (MLS) for training a piper-tts ONNX voice model.

## Usage

```bash
# 1. Identify the best female speaker
uv run analizar_mls.py

# 2. Export her clips as a LJSpeech dataset
uv run exportar_dataset.py --speaker <speaker_id>
```

The output (`./data/dataset/`) contains WAV files at 22050 Hz and a `metadata.csv` in LJSpeech format, ready to upload to Kaggle / Google Colab / VPS for piper training.

## Data Attribution

Audio data sourced from **Multilingual LibriSpeech** (Pratap et al., 2020), derived from LibriVox recordings.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

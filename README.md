# common-voice-bookbot

Tooling to download and prepare a female Spanish voice dataset from Multilingual LibriSpeech (MLS) for training a piper-tts ONNX voice model.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- [Task](https://taskfile.dev) *(optional, for shorthand commands)* — `winget install Task.Task`
- A HuggingFace token in `.env` as `HF_TOKEN=hf_...`

## Quick start

```bash
uv sync                                          # install dependencies
uv run analizar_mls.py                           # find best female speaker
uv run exportar_dataset.py --speaker <id>        # export her clips
```

Or with [Task](https://taskfile.dev):

```bash
task sync                        # install dependencies
task analizar                    # find best female speaker
task exportar SPEAKER=<id>       # export her clips
task test                        # run tests
task clean                       # delete ./data/ (asks confirmation)
```

## Workflow

### 1. Identify a speaker

```bash
task analizar
```

Prints the top 10 female speakers from MLS Spanish ranked by total minutes. Copy the `speaker_id` of the one you want.

```
Top hablantes femeninas en MLS Spanish (por minutos de audio):

#    speaker_id       minutos
--------------------------------
1    2138             127.4
2    5491              84.1
...

Para exportar la hablante con más audio:
  uv run exportar_dataset.py --speaker 2138
```

### 2. Export the dataset

```bash
task exportar SPEAKER=2138
```

Downloads all clips from that speaker, resamples to **22050 Hz mono**, normalizes loudness to **-23 LUFS (EBU R128)**, and writes:

```
./data/dataset/
  wavs/
    2138_0001.wav
    2138_0002.wav
    ...
  metadata.csv        ← LJSpeech format: wavs/<file>.wav|transcription
```

### 3. Train on the cloud

Upload `./data/dataset/` to Kaggle / Google Colab / VPS and run the [piper training scripts](https://github.com/rhasspy/piper/blob/master/TRAINING.md).

## Development

```bash
task test     # run all tests (16 tests)
task sync     # re-sync dependencies after pyproject.toml changes
```

## Data Attribution

Audio data sourced from **Multilingual LibriSpeech** (Pratap et al., 2020), derived from LibriVox recordings.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

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

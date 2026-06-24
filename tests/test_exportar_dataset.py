import io
from pathlib import Path

import numpy as np
import soundfile as sf

from exportar_dataset import build_wav_filename, decode_audio_bytes, write_metadata_csv


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


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 22050) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="wav", subtype="FLOAT")
    return buf.getvalue()


def test_decode_audio_bytes_returns_float32():
    audio, sr = decode_audio_bytes(_make_wav_bytes())
    assert audio.dtype == np.float32


def test_decode_audio_bytes_correct_samplerate():
    audio, sr = decode_audio_bytes(_make_wav_bytes(sr=22050))
    assert sr == 22050


def test_decode_audio_bytes_mono_output():
    audio, sr = decode_audio_bytes(_make_wav_bytes())
    assert audio.ndim == 1


def test_decode_audio_bytes_correct_duration():
    audio, sr = decode_audio_bytes(_make_wav_bytes(duration_s=1.5, sr=16000))
    assert sr == 16000
    assert abs(len(audio) / sr - 1.5) < 0.05  # within 50ms

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

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

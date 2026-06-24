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

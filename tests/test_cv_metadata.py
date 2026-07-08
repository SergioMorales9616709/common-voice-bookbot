from pathlib import Path

import pytest

TSV_CONTENT = (
    "client_id\tpath\tgender\tsentence\n"
    "abc123\tclips/001.mp3\tfemale_feminine\tHola mundo.\n"
    "abc123\tclips/002.mp3\tfemale_feminine\tEl cielo es azul.\n"
    "def456\tclips/003.mp3\tmale_masculine\tEl río corre.\n"
    "xyz789\tclips/004.mp3\tfemale_feminine\tLa casa.\n"
    "ghi000\tclips/005.mp3\t\tSin género declarado.\n"
)


def test_find_clips_tsv_finds_file_in_root(tmp_path: Path):
    tsv = tmp_path / "clips.tsv"
    tsv.write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import find_clips_tsv

    assert find_clips_tsv(tmp_path) == tsv


def test_find_clips_tsv_finds_file_in_subdirectory(tmp_path: Path):
    subdir = tmp_path / "corpus" / "es"
    subdir.mkdir(parents=True)
    tsv = subdir / "validated.tsv"
    tsv.write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import find_clips_tsv

    assert find_clips_tsv(tmp_path) == tsv


def test_find_clips_tsv_prefers_validated_over_other_splits(tmp_path: Path):
    (tmp_path / "dev.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    (tmp_path / "other.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    validated = tmp_path / "validated.tsv"
    validated.write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import find_clips_tsv

    assert find_clips_tsv(tmp_path) == validated


def test_find_clips_tsv_raises_when_not_found(tmp_path: Path):
    from cv_metadata import find_clips_tsv

    with pytest.raises(FileNotFoundError, match="descargar-cv"):
        find_clips_tsv(tmp_path)


def test_get_top_female_speakers_columns(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers

    df = get_top_female_speakers(tmp_path, n=10)
    assert list(df.columns) == ["client_id", "clips", "minutes"]


def test_get_top_female_speakers_excludes_males(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers

    df = get_top_female_speakers(tmp_path)
    assert "def456" not in df["client_id"].values


def test_get_top_female_speakers_sorted_desc(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers

    df = get_top_female_speakers(tmp_path)
    assert df.iloc[0]["client_id"] == "abc123"
    clips = df["clips"].tolist()
    assert clips == sorted(clips, reverse=True)


def test_get_top_female_speakers_minutes(tmp_path: Path):
    (tmp_path / "clips.tsv").write_text(TSV_CONTENT, encoding="utf-8")
    from cv_metadata import get_top_female_speakers

    df = get_top_female_speakers(tmp_path)
    # abc123 has 2 clips × 5.0s / 60 = 0.1667 min
    assert df.iloc[0]["clips"] == 2
    assert df.iloc[0]["minutes"] == pytest.approx(2 * 5.0 / 60)


def test_get_top_female_speakers_respects_n(tmp_path: Path):
    lines = ["client_id\tpath\tgender\tsentence"]
    for i in range(10):
        lines.append(f"speaker_{i:03d}\tclips/{i:03d}.mp3\tfemale_feminine\tText {i}")
    (tmp_path / "clips.tsv").write_text("\n".join(lines), encoding="utf-8")
    from cv_metadata import get_top_female_speakers

    df = get_top_female_speakers(tmp_path, n=3)
    assert len(df) == 3

import pandas as pd
import pytest
from mls_metadata import parse_metainfo, get_female_speakers

# Real format: pipe-separated, header row, one row per book chapter.
# Multiple rows per speaker — minutes must be aggregated.
SAMPLE_METAINFO = (
    " SPEAKER   |   GENDER   | PARTITION  |  MINUTES   |  BOOK ID   |  TITLE  |  CHAPTER  \n"
    "   2138    |     F      |   train    |   15.3     |   1001     |  Book A |  Chapter 1\n"
    "   2138    |     F      |   train    |   12.1     |   1001     |  Book A |  Chapter 2\n"
    "   3421    |     M      |   train    |    8.7     |   1002     |  Book B |  Chapter 1\n"
    "   5491    |     F      |   train    |   84.1     |   1003     |  Book C |  Chapter 1\n"
    "   6012    |     F      |   dev      |   12.3     |   1004     |  Book D |  Chapter 1\n"
)


def test_parse_metainfo_columns():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert list(df.columns) == ["speaker_id", "gender", "minutes"]


def test_parse_metainfo_speaker_ids_are_strings():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert pd.api.types.is_string_dtype(df["speaker_id"])


def test_parse_metainfo_aggregates_per_speaker():
    # 5 chapter rows → 4 unique speakers
    df = parse_metainfo(SAMPLE_METAINFO)
    assert len(df) == 4


def test_parse_metainfo_sums_minutes_across_chapters():
    df = parse_metainfo(SAMPLE_METAINFO)
    row = df[df["speaker_id"] == "2138"].iloc[0]
    assert row["gender"] == "F"
    assert row["minutes"] == pytest.approx(27.4)  # 15.3 + 12.1


def test_get_female_speakers_excludes_males():
    df = parse_metainfo(SAMPLE_METAINFO)
    females = get_female_speakers(df)
    assert all(females["gender"] == "F")
    assert "3421" not in females["speaker_id"].values


def test_get_female_speakers_sorted_desc():
    df = parse_metainfo(SAMPLE_METAINFO)
    females = get_female_speakers(df)
    minutes = females["minutes"].tolist()
    assert minutes == sorted(minutes, reverse=True)


# --- Integration tests (require internet + HF_TOKEN in .env) ---

@pytest.mark.integration
def test_download_speaker_metadata_returns_valid_dataframe():
    from mls_metadata import download_speaker_metadata
    df = download_speaker_metadata()
    assert not df.empty
    assert list(df.columns) == ["speaker_id", "gender", "minutes"]
    assert pd.api.types.is_string_dtype(df["speaker_id"])
    assert df["minutes"].dtype == float


@pytest.mark.integration
def test_download_speaker_metadata_has_female_speakers():
    from mls_metadata import download_speaker_metadata, get_female_speakers
    df = download_speaker_metadata()
    females = get_female_speakers(df)
    assert not females.empty
    assert females.iloc[0]["minutes"] > 0

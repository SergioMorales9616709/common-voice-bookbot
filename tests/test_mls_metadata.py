import pandas as pd
import pytest
from mls_metadata import parse_metainfo, get_female_speakers

SAMPLE_METAINFO = (
    "2138\ttrain\t127.4\tSome Name\tF\n"
    "3421\ttrain\t98.2\tAnother Name\tM\n"
    "5491\ttrain\t84.1\tThird Reader\tF\n"
    "6012\tdev\t12.3\tFourth Reader\tF\n"
)


def test_parse_metainfo_columns():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert list(df.columns) == ["speaker_id", "gender", "minutes"]


def test_parse_metainfo_speaker_ids_are_strings():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert df["speaker_id"].dtype == object


def test_parse_metainfo_row_count():
    df = parse_metainfo(SAMPLE_METAINFO)
    assert len(df) == 4


def test_parse_metainfo_values():
    df = parse_metainfo(SAMPLE_METAINFO)
    row = df[df["speaker_id"] == "2138"].iloc[0]
    assert row["gender"] == "F"
    assert row["minutes"] == pytest.approx(127.4)


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

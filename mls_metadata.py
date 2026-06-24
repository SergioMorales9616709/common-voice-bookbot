import io
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

_METAINFO_FILENAME = "data/mls_spanish/metainfo.txt"


def parse_metainfo(content: str) -> pd.DataFrame:
    df = pd.read_csv(
        io.StringIO(content),
        sep="|",
        dtype=str,
    )
    df.columns = [c.strip() for c in df.columns]
    df["SPEAKER"] = df["SPEAKER"].str.strip()
    df["GENDER"] = df["GENDER"].str.strip()
    df["MINUTES"] = pd.to_numeric(df["MINUTES"].str.strip(), errors="coerce")

    result = (
        df.groupby(["SPEAKER", "GENDER"], as_index=False)
        .agg(minutes=("MINUTES", "sum"))
        .rename(columns={"SPEAKER": "speaker_id", "GENDER": "gender"})
    )
    return result[["speaker_id", "gender", "minutes"]]


def get_female_speakers(df: pd.DataFrame) -> pd.DataFrame:
    females = df[df["gender"] == "F"].copy()
    return females.sort_values("minutes", ascending=False).reset_index(drop=True)


def download_speaker_metadata() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="facebook/multilingual_librispeech",
        filename=_METAINFO_FILENAME,
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return parse_metainfo(content)

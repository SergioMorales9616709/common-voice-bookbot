import io
import pandas as pd
from huggingface_hub import hf_hub_download


def parse_metainfo(content: str) -> pd.DataFrame:
    df = pd.read_csv(
        io.StringIO(content),
        sep="\t",
        header=None,
        names=["speaker_id", "subset", "minutes", "name", "gender"],
        dtype={"speaker_id": object},
    )
    return df[["speaker_id", "gender", "minutes"]]


def get_female_speakers(df: pd.DataFrame) -> pd.DataFrame:
    females = df[df["gender"] == "F"].copy()
    return females.sort_values("minutes", ascending=False).reset_index(drop=True)


def download_speaker_metadata() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="facebook/multilingual_librispeech",
        filename="spanish/metainfo.txt",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return parse_metainfo(content)

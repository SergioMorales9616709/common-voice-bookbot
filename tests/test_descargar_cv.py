import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_tarball(tmp_path: Path, files: dict[str, bytes]) -> Path:
    archive = tmp_path / "corpus.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return archive


def test_verify_checksum_correct(tmp_path: Path):
    content = b"corpus data"
    f = tmp_path / "file.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    from descargar_cv import verify_checksum

    assert verify_checksum(f, expected) is True


def test_verify_checksum_wrong(tmp_path: Path):
    content = b"corpus data"
    f = tmp_path / "file.bin"
    f.write_bytes(content)
    from descargar_cv import verify_checksum

    assert verify_checksum(f, "deadbeef") is False


def test_extract_archive_creates_files(tmp_path: Path):
    archive = _make_tarball(
        tmp_path,
        {
            "corpus/es/clips.tsv": b"client_id\tpath\n",
            "corpus/es/clips/001.mp3": b"fakeaudio",
        },
    )
    output_dir = tmp_path / "output"
    from descargar_cv import extract_archive

    extract_archive(archive, output_dir)
    assert (output_dir / "corpus" / "es" / "clips.tsv").exists()
    assert (output_dir / "corpus" / "es" / "clips" / "001.mp3").exists()


def test_get_download_url_returns_dict():
    fake_response = {
        "downloadUrl": "https://storage.example.com/corpus.tar.gz",
        "sizeBytes": 48_000_000_000,
        "checksum": "abc123def456",
        "filename": "common-voice-scripted-speech-26-0-es.tar.gz",
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status = MagicMock()

    with patch("descargar_cv.requests.post", return_value=mock_resp) as mock_post:
        from descargar_cv import get_download_url

        result = get_download_url("my-api-key")

    assert result["downloadUrl"] == fake_response["downloadUrl"]
    assert result["sizeBytes"] == fake_response["sizeBytes"]
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-api-key"


def test_get_download_url_raises_on_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch("descargar_cv.requests.post", return_value=mock_resp):
        from descargar_cv import get_download_url

        with pytest.raises(Exception, match="401"):
            get_download_url("bad-key")


def test_download_file_sends_range_header_on_resume(tmp_path: Path):
    part_file = tmp_path / "file.part"
    already_downloaded = b"first_chunk"
    part_file.write_bytes(already_downloaded)

    new_chunk = b"_second_chunk"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 206  # server accepts range request
    mock_resp.iter_content.return_value = [new_chunk]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured = {}

    def fake_get(url, headers=None, stream=None, timeout=None):
        captured["headers"] = headers or {}
        return mock_resp

    with patch("descargar_cv.requests.get", side_effect=fake_get):
        from descargar_cv import download_file

        download_file("https://storage.example.com/file", part_file, 100)

    assert captured["headers"].get("Range") == f"bytes={len(already_downloaded)}-"
    assert part_file.read_bytes() == already_downloaded + new_chunk


def test_download_file_no_range_header_on_fresh_start(tmp_path: Path):
    part_file = tmp_path / "file.part"

    chunk = b"full_content"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200  # fresh start doesn't need 206
    mock_resp.iter_content.return_value = [chunk]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    captured = {}

    def fake_get(url, headers=None, stream=None, timeout=None):
        captured["headers"] = headers or {}
        return mock_resp

    with patch("descargar_cv.requests.get", side_effect=fake_get):
        from descargar_cv import download_file

        download_file("https://storage.example.com/file", part_file, 100)

    assert "Range" not in captured["headers"]
    assert part_file.read_bytes() == chunk


def test_download_file_raises_when_resume_gets_200(tmp_path: Path):
    part_file = tmp_path / "file.part"
    part_file.write_bytes(b"existing_data")  # simulate partial download

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200  # server ignores Range header
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("descargar_cv.requests.get", return_value=mock_resp):
        from descargar_cv import download_file

        with pytest.raises(RuntimeError, match="206"):
            download_file("https://example.com/file", part_file, 100)


# --- Integration test (requires MDC_API_KEY in .env + internet) ---


@pytest.mark.integration
def test_get_download_url_integration():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    from descargar_cv import get_download_url

    result = get_download_url(os.environ["MDC_API_KEY"])
    assert "downloadUrl" in result
    assert result["sizeBytes"] > 0

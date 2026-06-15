from pathlib import Path

from secure_rag.loaders import load_path


def test_load_csv_creates_one_document_per_record() -> None:
    documents = load_path(Path("tests/fixtures/security_controls.csv"))

    assert len(documents) == 3
    assert "control_id: AC-001" in documents[0].text
    assert "Require multi-factor authentication" in documents[0].text
    assert documents[0].metadata["record_index"] == 1
    assert documents[0].metadata["file_type"] == "csv"


def test_load_directory_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ransomware backup control", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"nope")

    documents = load_path(tmp_path)

    assert len(documents) == 1
    assert documents[0].text == "ransomware backup control"


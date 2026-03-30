from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NAO_DIR = ROOT / "nld-nao"
NAO_ZIP_PATH = NAO_DIR / "nld-nao-dir-aa.zip"
NAO_EXTRACTED_ROOT = NAO_DIR / "nld-nao-unzipped"


class HumanNAODataSource:
    def __init__(self) -> None:
        self.base_dir = NAO_EXTRACTED_ROOT.parent
        self.extracted_root = NAO_EXTRACTED_ROOT
        self.zip_path = NAO_ZIP_PATH
        self.mode = "dir" if self.extracted_root.exists() else "zip"
        self._zip: zipfile.ZipFile | None = None

    def __enter__(self) -> HumanNAODataSource:
        if self.mode == "zip":
            self._zip = zipfile.ZipFile(self.zip_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def namelist(self) -> list[str]:
        if self.mode == "dir":
            return [
                path.relative_to(self.base_dir).as_posix()
                for path in self.extracted_root.rglob("*")
                if path.is_file()
            ]
        if self._zip is None:
            raise RuntimeError("Zip source is not open.")
        return self._zip.namelist()

    def open(self, member_name: str):
        if self.mode == "dir":
            return (self.base_dir / Path(member_name)).open("rb")
        if self._zip is None:
            raise RuntimeError("Zip source is not open.")
        return self._zip.open(member_name)



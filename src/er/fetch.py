"""Download the benchmark archives and record exactly what was downloaded.

The archives are small, but provenance still matters: these files have been mirrored in a
dozen places with silently different contents, so the hash and the retrieval date are recorded
alongside the data and printed into the report. If a future run produces a different hash, that
is a change in the source, not a change in the code, and the difference should be visible.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_RAW, load_config


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(key: str, spec: dict, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{key}.zip"
    if not archive.exists():
        urllib.request.urlretrieve(spec["url"], archive)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        zf.extractall(dest)
    return {
        "dataset": key,
        "url": spec["url"],
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_of(archive),
        "members": sorted(members),
    }


def main() -> None:
    config = load_config()
    provenance = []
    for key, spec in config["datasets"].items():
        record = fetch_one(key, spec, DATA_RAW / key)
        provenance.append(record)
        print(f"{spec['label']:24s} {record['archive_bytes']:>9,} bytes  "
              f"sha256={record['archive_sha256'][:16]}...")
    out = DATA_RAW / "provenance.json"
    out.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nprovenance written to {out.relative_to(out.parents[2])}")


if __name__ == "__main__":
    main()

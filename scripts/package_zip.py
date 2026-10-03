"""Make a ZIP alongside the package, excluding transient Python bytecode."""

from pathlib import Path
import json
import zipfile

PACK = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((PACK / "qa/validation.json").read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise RuntimeError("Run verify_bundle.py and resolve issues before packaging")
    if not (PACK / "MANIFEST_SHA256.json").is_file():
        raise FileNotFoundError("Run verify_bundle.py --write-manifest before packaging")
    dest = PACK.parent / f"{PACK.name}.zip"
    count = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PACK.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, Path(PACK.name) / path.relative_to(PACK))
                count += 1
    with zipfile.ZipFile(dest) as archive:
        damaged = archive.testzip()
        if damaged:
            raise RuntimeError(f"ZIP CRC mismatch: {damaged}")
    print(f"{dest}\n{count} files; {dest.stat().st_size / 1024**2:.2f} MiB; CRC verified")


if __name__ == "__main__":
    main()

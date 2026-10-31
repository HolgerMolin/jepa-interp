"""Download the pinned checkpoint to scratch, resume transfers, verify SHA-256."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "checkpoint_manifest.json").read_text())
dest = Path(manifest["destination"])
dest.mkdir(parents=True, exist_ok=True)
for entry in manifest["files"]:
    name = entry["rfilename"]
    target = dest / name
    partial = dest / (name + ".partial")
    if not target.exists():
        url = f'https://huggingface.co/{manifest["repo_id"]}/resolve/{manifest["revision"]}/{name}'
        print(f"Downloading {name} ({entry['size']:,} bytes)", flush=True)
        subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5",
                        "--connect-timeout", "30", "-C", "-", "-o", str(partial), url], check=True)
        candidate = partial
    else:
        candidate = target
    if candidate.stat().st_size != entry["size"]:
        raise RuntimeError(f"Size mismatch: {candidate}")
    if "lfs" in entry:
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["lfs"]["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {candidate}")
    if candidate != target:
        candidate.rename(target)
    print(f"Verified {name}", flush=True)
(dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"COMPLETE: {dest}", flush=True)

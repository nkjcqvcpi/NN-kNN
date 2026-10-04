"""Download and hash the explicitly pinned small T3 engineering host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Exact 40-character Qwen3-0.6B commit")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("revision must be an immutable commit, not main or a branch")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    repo = "Qwen/Qwen3-0.6B"
    info = HfApi(token=False).model_info(repo, revision=args.revision)
    if info.sha != args.revision or info.private or info.gated:
        raise ValueError("host must resolve to the exact declared public, ungated checkpoint")
    names = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "model.safetensors", "README.md", "LICENSE"}
    if args.model_dir.exists():
        extra = {p.name for p in args.model_dir.iterdir() if p.is_file()} - names
        if extra:
            raise ValueError("model directory contains files outside this host contract; use a separate directory")
    snapshot_download(repo, revision=args.revision, local_dir=args.model_dir,
                      token=False, allow_patterns=sorted(names))
    records = []
    for path in sorted(args.model_dir.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    if {r["name"] for r in records} != names:
        raise ValueError("the declared host snapshot is incomplete")
    report = {"model": repo, "revision": args.revision, "private": False, "gated": False,
              "model_card": f"https://huggingface.co/{repo}/blob/{args.revision}/README.md",
              "declared_license": "apache-2.0", "files": records,
              "scope": "small local engineering host; no benchmark evidence"}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model": repo, "revision": args.revision, "file_count": len(records)}))


if __name__ == "__main__":
    main()

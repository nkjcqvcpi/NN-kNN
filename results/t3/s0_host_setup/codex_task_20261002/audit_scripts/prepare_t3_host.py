import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import json, hashlib
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

repo, revision = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
info = HfApi(token=False).model_info(repo, revision=revision)
assert info.sha == revision and not info.private and not info.gated
target = Path(__file__).parent / "t3-models" / "qwen3-0.6b"
snapshot_download(repo, revision=revision, local_dir=target, token=False,
    allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "vocab.json", "merges.txt", "model.safetensors", "README.md", "LICENSE"])
files = []
for path in sorted(target.iterdir()):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                digest.update(chunk)
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
assert any(f["name"] == "model.safetensors" and f["bytes"] > 100_000_000 for f in files)
manifest = {"model": repo, "revision": revision, "private": False, "gated": False,
    "model_card": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
    "declared_license": "apache-2.0", "scope": "small local engineering host, no benchmark result yet", "files": files}
(Path(__file__).parent / "t3-host-files.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"model": repo, "revision": revision, "file_count": len(files), "weights_bytes": next(f["bytes"] for f in files if f["name"] == "model.safetensors")}))

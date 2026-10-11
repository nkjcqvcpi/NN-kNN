from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

repo = Path(r"C:\Users\Administrator\NN-KNN\nnknn-work")
work = Path(__file__).resolve().parent
archive = work / "t3_prompt_pilot_source_archive"
archive.mkdir(exist_ok=False)
digest = hashlib.sha256()
paths = []
for directory in ("model", "tools", "configs", "tests", "datasets"):
    for path in sorted((repo/directory).rglob("*")):
        if path.suffix in {".py", ".yaml", ".ps1"} and "__pycache__" not in path.parts:
            relative = path.relative_to(repo).as_posix()
            content = path.read_bytes()
            digest.update(relative.encode())
            digest.update(content)
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            paths.append((relative, content))
fingerprint = digest.hexdigest()
zip_path = work / f"t3_prompt_pilot_source_{fingerprint[:12]}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
    for name, content in paths:
        bundle.writestr(name, content)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "model", "tools", "configs", "tests", "datasets"], cwd=repo, text=True)
(work / "t3_prompt_pilot_git_status_before.txt").write_text(dirty, encoding="utf-8")
metadata = dict(fingerprint=fingerprint, head=head, entries=len(paths), archive=str(archive),
    zip=str(zip_path), zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest())
(work / "t3_prompt_pilot_freeze.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata))

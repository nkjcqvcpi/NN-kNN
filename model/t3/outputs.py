"""Where T3 tools write generated output (owner, 2026-11-22: O2/O3).

Generated output goes under ``<worktree>/results/t3/<stage>/`` (git-ignored). ``reports/`` and ``docs/`` hold only
reports written by agents or people, so a tool refuses an output path inside either of them.
"""
from datetime import datetime,timezone
from pathlib import Path

WORKTREE=Path(__file__).resolve().parents[2]
RESULTS=WORKTREE/'results'/'t3'
TRACKED=('reports','docs')


def refuse_tracked_output(path):
    """Raise ValueError if ``path`` lies inside this worktree's reports/ or docs/."""
    resolved=Path(path).resolve()
    for name in TRACKED:
        folder=(WORKTREE/name).resolve()
        if resolved==folder or folder in resolved.parents:
            raise ValueError(f'{path} is inside {name}/, which holds only written reports; generated output goes under results/')
    return Path(path)


def default_output(stage,name,suffix=''):
    """A fresh default path ``results/t3/<stage>/<name>-<UTC timestamp><suffix>`` (a folder,or a file with ``suffix``)."""
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return RESULTS/stage/f'{name}-{stamp}{suffix}'


def output_path(path,stage,name,suffix=''):
    """The output a tool writes: ``path`` if given,else the default under results/; never inside reports/ or docs/."""
    return refuse_tracked_output(default_output(stage,name,suffix) if path is None else path)

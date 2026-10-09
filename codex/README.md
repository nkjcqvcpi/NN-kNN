# Branch validation

Branch: `rl`. See `../README.md` and `../docs/BRANCH_ORGANIZATION.md`.

Run `python codex/smoke_test.py --mode imports`; available training modes are shown by `--help`. Run `python -m pytest tests -q --basetemp=<new-workspace-temp-directory>`. Use `bash codex/setup.sh` for a fresh CPU cloud environment.

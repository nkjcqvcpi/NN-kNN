# Local frozen-host engineering preparation — 2026-10-04

The T3 specification requires an existing host, original artifact reuse rather
than NN-CDH memory adaptation, bounded observable requests, shared host/audit
retrieval events, scoped typed memories, and eventual prompt/internal/combined
comparisons on the same host checkpoint. This package establishes only a local
host prerequisite. None of those T3 integrations, comparisons, biomedical
benchmarks or competitive agent results is complete.

The small engineering host is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B),
fixed at `c1899de289a04d12100db370d81485cdf75e47ca`. Its official card declares
Apache-2.0 and Transformers >=4.51 support. This is an exploratory engineering
choice, not a PI-approved confirmatory host or substitute for the planned cloud
and stronger-model external-validity studies. All host parameters are frozen.

## Reproducible tools and artifacts

`tools/t3_prepare_host.py` anonymously resolves an exact immutable public,
ungated commit, downloads the nine declared model/tokenizer/license/card files,
and writes file sizes and SHA256 hashes. It rejects branch names, incomplete
snapshots and unrelated direct files. `tools/t3_host_smoke.py` checks the expected
revision and every local file against that manifest before loading. No remote
model code executes. The smoke records driver hash, environment, hardware,
decoding settings, full observable prompt/output, token counts, timings and
memory. These are two repeated generations, not benchmark cases.

The prepared `model.safetensors` contains 1,503,300,328 bytes and SHA256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The actual instantiated host has 596,049,920 parameters and zero trainable
parameters. The local Intel Arc A770/XPU run uses float16 and eager attention,
with about 1.20 GB peak tensor allocation. It produces `READY.` twice with
identical tokens; actual generation-step scores are finite. This verifies local
loading/generation and fixed decoding, not answer quality or retrieval benefit.

The host environment is separate from the T1 training environment, with
Transformers 4.57.6, Hugging Face Hub 0.36.2 and safetensors 0.8.0, exposing the
existing PyTorch 2.14.0+xpu runtime. Windows requires the existing native
`Library/bin` directory in this process's DLL search/PATH. An initial attempt
failed with WinError 126 when only Python package paths were exposed; no global
machine configuration was changed. The CLI provides `--runtime-library-bin`.

An initial fresh `GenerationConfig(do_sample=False)` was overwritten by
checkpoint-specific sampling defaults in this library version. That attempt is
preserved in the task work folder and is not deterministic evidence. The final
driver passes `do_sample=False` explicitly to `generate`, disables thinking and
clears irrelevant sampling settings. Repeated token equality is checked, with
the actual per-step scores tested for finite values. Private reasoning is not
requested or recorded.

Example, inside the optional host environment:

```powershell
python tools/t3_prepare_host.py --revision c1899de289a04d12100db370d81485cdf75e47ca --model-dir <local-model-dir> --manifest <local-files-manifest.json>
python tools/t3_host_smoke.py --model-dir <local-model-dir> --files-manifest <local-files-manifest.json> --expected-revision c1899de289a04d12100db370d81485cdf75e47ca --device xpu --runtime-library-bin <torch-runtime-Library-bin> --out <host-smoke.json>
```

The later scoped prompt engineering fixture is documented separately in
`T3_PROMPT_ENGINEERING_PILOT.md`: actual host requests and original-evidence
reuse now run, while internal integration and public benchmark evidence remain
open. The original smoke results above retain their prerequisite-only boundary.

The task outputs contain the smoke JSON; model weights, package environment,
download manifest and rejected preliminary attempt remain in its work folder.
Keep weights outside Git. CPU is the portable default; use an explicitly
available accelerator and record it. Next work must implement scoped typed
retrieval and event fidelity, bounded host requests and actual downstream tests
before calling this an integrated T3 system. Internal interface and trainable
scope remain explicitly exploratory until tested and frozen.

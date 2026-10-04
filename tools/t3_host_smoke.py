"""Validate a pinned local open-weight host; this is not a T3 benchmark.

Use a separate optional Transformers environment. It can expose the existing
Torch installation and use --runtime-library-bin for its Windows native runtime.
No weights are downloaded here and no remote model code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--files-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--runtime-library-bin", type=Path)
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="cpu")
    parser.add_argument("--prompt", default="Reply with only the word READY.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        parser.error("engineering smoke token budget must be within 1..128")
    native_handle = None
    if args.runtime_library_bin is not None:
        native = args.runtime_library_bin.resolve(strict=True)
        os.environ["PATH"] = str(native) + os.pathsep + os.environ.get("PATH", "")
        if sys.platform == "win32":
            native_handle = os.add_dll_directory(str(native))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads(args.files_manifest.read_text(encoding="utf-8"))
    if manifest["revision"] != args.expected_revision:
        raise ValueError("local manifest revision differs from the declared host checkpoint")
    names = set()
    for record in manifest["files"]:
        name = record["name"]
        if Path(name).name != name or name in names:
            raise ValueError("manifest file names must be unique direct-child names")
        names.add(name)
        path = args.model_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"pinned host file failed integrity check: {name}")
    if not {"config.json", "tokenizer_config.json", "model.safetensors"} <= names:
        raise ValueError("manifest must cover the actual model and tokenizer configuration")
    if {p.name for p in args.model_dir.iterdir() if p.is_file()} != names:
        raise ValueError("all direct model/tokenizer files must be covered by the pinned manifest")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32 if args.device == "cpu" else torch.float16,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    load_seconds = time.perf_counter() - start
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    # Pass the control explicitly to generate: fresh GenerationConfig defaults
    # can otherwise be replaced with checkpoint-specific sampling defaults.
    decode = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None, eos_token_id=model.config.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, use_cache=True)
    accelerator = getattr(torch, args.device) if args.device in {"xpu", "cuda"} else None
    if accelerator is not None:
        accelerator.reset_peak_memory_stats()
    def synchronize():
        if accelerator is not None:
            accelerator.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, **decode, return_dict_in_generate=True, output_scores=True)
        synchronize()
        first_seconds = time.perf_counter() - start
        repeated = model.generate(**inputs, **decode)
        synchronize()
    output = generated.sequences
    finite = bool(generated.scores) and all(bool(torch.isfinite(s).all()) for s in generated.scores)
    repeated_equal = torch.equal(output, repeated)
    if not finite or not repeated_equal:
        raise ValueError("host smoke requires finite actual generation scores and repeatable greedy tokens")
    report = dict(scope="two local frozen-host engineering generations; no T3 integration or benchmark result",
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version.split()[0],
        model=manifest["model"], revision=manifest["revision"], verified_files=len(names),
        declared_license=manifest.get("declared_license"), model_card=manifest.get("model_card"),
        device=args.device, hardware=accelerator.get_device_name() if accelerator else "CPU",
        dtype=str(next(model.parameters()).dtype), torch=torch.__version__,
        transformers=transformers.__version__, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        enable_thinking=False, prompt=prompt,
        answer=tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True),
        input_tokens=inputs["input_ids"].shape[1], output_tokens=output.shape[1]-inputs["input_ids"].shape[1],
        decode_parameters=decode, load_seconds=load_seconds, first_generation_seconds=first_seconds,
        two_generation_seconds=time.perf_counter()-start, greedy_repeat_token_equal=repeated_equal,
        generated_step_scores_finite=finite,
        peak_allocated_bytes=accelerator.max_memory_allocated() if accelerator else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "revision", "device", "answer",
        "verified_files", "generated_step_scores_finite", "greedy_repeat_token_equal")}))


if __name__ == "__main__":
    main()

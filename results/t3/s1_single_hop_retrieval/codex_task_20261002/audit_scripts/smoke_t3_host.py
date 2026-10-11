import os, json, time
from pathlib import Path
import sys
os.environ["PATH"] = r"C:\Users\Administrator\NN-KNN\nnknn-work\.venv\Library\bin" + os.pathsep + os.environ["PATH"]
task_dll_handle = os.add_dll_directory(r"C:\Users\Administrator\NN-KNN\nnknn-work\.venv\Library\bin")
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

root = Path(__file__).parent
model_path = root / "t3-models/qwen3-0.6b"
torch.manual_seed(0)
torch.set_num_threads(2)
device = "xpu" if torch.xpu.is_available() else "cpu"
start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
    trust_remote_code=False, dtype=torch.float16 if device == "xpu" else torch.float32,
    attn_implementation="eager").to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
load_seconds = time.perf_counter() - start
prompt = "Reply with only the word READY."
text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(device)
decode = dict(max_new_tokens=16, do_sample=False, temperature=1., top_p=1., top_k=0,
              eos_token_id=model.config.eos_token_id, pad_token_id=tokenizer.eos_token_id, use_cache=True)
if device == "xpu":
    torch.xpu.reset_peak_memory_stats()
start = time.perf_counter()
with torch.no_grad():
    generated = model.generate(**inputs, **decode, return_dict_in_generate=True, output_scores=True)
    repeated = model.generate(**inputs, **decode)
output = generated.sequences
if device == "xpu":
    torch.xpu.synchronize()
generation_seconds = time.perf_counter() - start
answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
report = {"scope": "one local frozen-host engineering smoke, no T3 integration or benchmark result",
    "model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "device": device, "dtype": str(next(model.parameters()).dtype),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "enable_thinking": False, "prompt": text, "answer": answer,
    "input_tokens": inputs["input_ids"].shape[1], "output_tokens": output.shape[1]-inputs["input_ids"].shape[1],
    "load_seconds": load_seconds, "generation_seconds": generation_seconds,
    "peak_allocated_bytes": torch.xpu.max_memory_allocated() if device == "xpu" else None,
    "decode_parameters": decode,
    "greedy_repeat_token_equal": torch.equal(output, repeated),
    "generated_step_scores_finite": all(bool(torch.isfinite(s).all()) for s in generated.scores)}
target = root.parent / "outputs/neural-cbr-t3-host-smoke-20261004.json"
target.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))

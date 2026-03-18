#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-14B with Transformers on ROCm (MI210)."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--prompt",
        default="Explain in 3 concise bullets what Slurm does in an HPC cluster.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="If set, run inference on prompts from this Hugging Face dataset.",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to run when --dataset-id is provided.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row index within the split when --dataset-id is provided.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/qwen3_14b_dataset_inference.jsonl",
        help="Output path for dataset mode.",
    )
    parser.add_argument(
        "--copy-columns",
        default="conflict,approach,intensity,context",
        help="Comma-separated dataset columns copied to each output row.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default="balanced",
        help="Use 'balanced' to spread layers across available GPUs.",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        default="52GiB",
        help="Per-GPU limit used by Accelerate device_map planner.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def build_inputs(tokenizer, prompt: str, device: torch.device):
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    encoded = tokenizer([text], return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def generate_one(model, tokenizer, args, prompt: str, device: torch.device):
    inputs = build_inputs(tokenizer, prompt, device)
    do_sample = args.temperature > 0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else None,
        "top_p": args.top_p if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    gen_s = time.time() - t0

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, gen_s


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm/CUDA device visible to PyTorch. "
            "Check Slurm GPU allocation and ROCm PyTorch install."
        )

    torch.manual_seed(args.seed)
    num_gpus = torch.cuda.device_count()

    print("=== Runtime info ===")
    print(f"Model: {args.model_id}")
    print(f"Visible GPUs: {num_gpus}")
    print(f"Torch version: {torch.__version__}")
    print(f"ROCm version: {torch.version.hip}")
    for idx in range(num_gpus):
        print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    print()

    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    max_memory = {gpu_idx: args.max_memory_per_gpu for gpu_idx in range(num_gpus)}
    max_memory["cpu"] = "128GiB"

    print("Loading model (this can take a while the first time)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s")

    # With model sharding, place inputs on the first model shard device.
    first_device = next(model.parameters()).device
    if args.dataset_id:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for --dataset-id mode. Install with `pip install datasets`."
            ) from exc

        if args.start_index < 0:
            raise ValueError("--start-index must be >= 0")
        if args.num_prompts <= 0:
            raise ValueError("--num-prompts must be > 0")

        dataset = load_dataset(args.dataset_id, split=args.dataset_split)
        total_rows = len(dataset)
        if args.prompt_column not in dataset.column_names:
            raise ValueError(
                f"Prompt column '{args.prompt_column}' not found. "
                f"Available columns: {dataset.column_names}"
            )
        if args.start_index >= total_rows:
            raise ValueError(
                f"--start-index ({args.start_index}) must be < dataset size ({total_rows})"
            )

        end_index = min(args.start_index + args.num_prompts, total_rows)
        copy_cols = [col.strip() for col in args.copy_columns.split(",") if col.strip()]
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"Running dataset inference: {args.dataset_id} [{args.dataset_split}] "
            f"rows {args.start_index}:{end_index} -> {out_path}"
        )
        started = time.time()
        with out_path.open("w", encoding="utf-8") as f:
            for i, ds_idx in enumerate(range(args.start_index, end_index), start=1):
                row = dataset[ds_idx]
                prompt = str(row[args.prompt_column])
                response, gen_s = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    prompt=prompt,
                    device=first_device,
                )

                record = {
                    "dataset_id": args.dataset_id,
                    "dataset_split": args.dataset_split,
                    "row_index": ds_idx,
                    "model_id": args.model_id,
                    "prompt": prompt,
                    "response": response,
                    "gen_seconds": round(gen_s, 3),
                }
                for col in copy_cols:
                    if col in row:
                        record[col] = row[col]

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{i}/{end_index - args.start_index}] row={ds_idx} gen={gen_s:.1f}s")

        print(f"Completed in {time.time() - started:.1f}s")
        print(f"Saved: {out_path.resolve()}")
    else:
        print("Generating...")
        response, gen_s = generate_one(
            model=model,
            tokenizer=tokenizer,
            args=args,
            prompt=args.prompt,
            device=first_device,
        )
        print(f"Generation finished in {gen_s:.1f}s")
        print("\n=== Prompt ===")
        print(args.prompt)
        print("\n=== Response ===")
        print(response)


if __name__ == "__main__":
    main()

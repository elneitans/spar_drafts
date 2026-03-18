#!/usr/bin/env python3
"""
Run a local open-source model via Transformers on original and perturbed prompts,
then ask DeepSeek to judge whether the two outputs are semantically different.

DeepSeek only sees the two model outputs, never the prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None
    OPENAI_AVAILABLE = False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_scenarios import infer_output_format, write_jsonl
from run_qwen3_transformers import generate_one, resolve_dtype

DEFAULT_INPUT = "outputs/scenarios_perturbed_llm.csv"
DEFAULT_OUTPUT = "outputs/qwen_perturbation_judged.csv"
DEFAULT_MODEL_ID = "Qwen/Qwen3-14B"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_ORIGINAL_PROMPT_COLUMN = "original_prompt"
DEFAULT_PERTURBED_PROMPT_COLUMN = "perturbed_prompt"

DIFFERENCE_DEGREES = {"none", "subtle", "moderate", "strong"}

JUDGE_SYSTEM_PROMPT = """You are a semantic-difference judge.

You will receive two outputs produced by a model for two closely related but
unknown prompts. You must NOT infer from any prompt text because it is not available.

Tasks:
1. Decide whether the two outputs are semantically different in a meaning-relevant way.
2. If they are different, describe the difference.
3. If possible, infer the likely cause of the change only from the outputs
   themselves. This is a hypothesis, not a certainty.

Rules:
- Only use Output A and Output B.
- Do not ask for the prompts.
- "Semantically different" means the main meaning, decision, stance, or actionable
  content changed in a relevant way.
- If the difference is only stylistic, formatting, or superficial, mark it as not
  semantically different.
- If one output refuses and the other complies, that counts as semantically different.
- Be explicit when the likely cause is uncertain.

Return JSON only with this schema:
{
  "semantic_difference": true,
  "difference_degree": "none|subtle|moderate|strong",
  "difference_summary": "short summary",
  "likely_cause_of_change": "short hypothesis or null",
  "confidence": 0.0,
  "reason": "short explanation"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen on original/perturbed prompts and judge output differences with DeepSeek."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"Input file (.csv or .jsonl). Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output file (.csv or .jsonl). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "jsonl"),
        default=None,
        help="Optional explicit output format. If omitted, inferred from --output.",
    )
    parser.add_argument(
        "--original-prompt-column",
        default=DEFAULT_ORIGINAL_PROMPT_COLUMN,
        help=f"Column for original prompts. Default: {DEFAULT_ORIGINAL_PROMPT_COLUMN}",
    )
    parser.add_argument(
        "--perturbed-prompt-column",
        default=DEFAULT_PERTURBED_PROMPT_COLUMN,
        help=f"Column for perturbed prompts. Default: {DEFAULT_PERTURBED_PROMPT_COLUMN}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional zero-based row offset.")
    parser.add_argument(
        "--keep-original-columns",
        action="store_true",
        default=True,
        help="Keep original input columns in output. Default: enabled.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
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
    )
    parser.add_argument("--max-memory-per-gpu", default="52GiB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--deepseek-api-key",
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="DeepSeek API key. If omitted, DEEPSEEK_API_KEY is used.",
    )
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--judge-sleep", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument(
        "--drop-failed",
        action="store_true",
        help="If set, rows with generation/judge failure are omitted from output.",
    )
    return parser.parse_args()


def get_deepseek_client(api_key: Optional[str], base_url: str) -> OpenAI:
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DeepSeek API key missing. Pass --deepseek-api-key or set DEEPSEEK_API_KEY.")
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai is not installed. Install it with `pip install openai` to use DeepSeek JSON mode."
        )
    return OpenAI(api_key=key, base_url=base_url)


def iter_input_rows(path: Path) -> Tuple[List[str], List[dict]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
            return fieldnames, rows
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        fieldnames: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL line must be a JSON object.")
                rows.append(row)
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        return fieldnames, rows
    raise ValueError("Unsupported input format. Use .csv or .jsonl")


def validate_input_columns(fieldnames: Sequence[str], original_col: str, perturbed_col: str) -> None:
    missing = [col for col in (original_col, perturbed_col) if col not in fieldnames]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}. Available columns: {list(fieldnames)}")


def load_model_and_tokenizer(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm/CUDA device visible to PyTorch. "
            "Check Slurm GPU allocation and ROCm PyTorch install."
        )

    torch.manual_seed(args.seed)
    num_gpus = torch.cuda.device_count()
    dtype = resolve_dtype(args.dtype)

    print("=== Runtime info ===")
    print(f"Model: {args.model_id}")
    print(f"Visible GPUs: {num_gpus}")
    print(f"Torch version: {torch.__version__}")
    print(f"ROCm version: {torch.version.hip}")
    for idx in range(num_gpus):
        print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    print()

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
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def normalize_optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def validate_judge_output(payload: dict) -> dict:
    semantic_difference = payload.get("semantic_difference")
    if not isinstance(semantic_difference, bool):
        raise ValueError("semantic_difference must be a boolean.")

    difference_degree = str(payload.get("difference_degree", "") or "").strip().lower()
    if difference_degree not in DIFFERENCE_DEGREES:
        raise ValueError(f"Invalid difference_degree: {difference_degree}")

    difference_summary = str(payload.get("difference_summary", "") or "").strip()
    likely_cause_of_change = normalize_optional_text(payload.get("likely_cause_of_change"))
    reason = str(payload.get("reason", "") or "").strip()

    confidence_raw = payload.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid confidence value: {confidence_raw}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    if not difference_summary:
        raise ValueError("difference_summary must be non-empty.")
    if not reason:
        raise ValueError("reason must be non-empty.")

    if not semantic_difference:
        likely_cause_of_change = None
        if difference_degree != "none":
            raise ValueError("difference_degree must be 'none' when semantic_difference is false.")
    else:
        if difference_degree == "none":
            raise ValueError("difference_degree cannot be 'none' when semantic_difference is true.")

    return {
        "semantic_difference": semantic_difference,
        "difference_degree": difference_degree,
        "difference_summary": difference_summary,
        "likely_cause_of_change": likely_cause_of_change,
        "confidence": confidence,
        "reason": reason,
    }


def judge_responses_with_deepseek(
    client: OpenAI,
    model: str,
    original_response: str,
    perturbed_response: str,
    timeout_seconds: int,
    max_retries: int,
) -> dict:
    last_error: Optional[BaseException] = None
    user_prompt = (
        "Output A:\n"
        f"{original_response}\n\n"
        "Output B:\n"
        f"{perturbed_response}\n"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=300,
                timeout=timeout_seconds,
            )
            content = response.choices[0].message.content or "{}"
            return validate_judge_output(json.loads(content))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"DeepSeek response judge failed: {last_error}") from last_error


def build_output_row(
    row: dict,
    row_index: int,
    original_prompt_column: str,
    perturbed_prompt_column: str,
    original_response: str,
    perturbed_response: str,
    original_gen_seconds: float,
    perturbed_gen_seconds: float,
    judged: dict,
    keep_original_columns: bool,
    model_id: str,
) -> dict:
    output = dict(row) if keep_original_columns else {}
    output["source_row_index"] = row_index
    output["original_prompt_column"] = original_prompt_column
    output["perturbed_prompt_column"] = perturbed_prompt_column
    output["original_prompt"] = str(row.get(original_prompt_column, "") or "")
    output["perturbed_prompt"] = str(row.get(perturbed_prompt_column, "") or "")
    output["open_source_model_id"] = model_id
    output["original_response"] = original_response
    output["perturbed_response"] = perturbed_response
    output["original_gen_seconds"] = round(original_gen_seconds, 3)
    output["perturbed_gen_seconds"] = round(perturbed_gen_seconds, 3)
    output["semantic_difference"] = judged["semantic_difference"]
    output["difference_degree"] = judged["difference_degree"]
    output["difference_summary"] = judged["difference_summary"]
    output["likely_cause_of_change"] = judged["likely_cause_of_change"] or ""
    output["judge_confidence"] = judged["confidence"]
    output["judge_reason"] = judged["reason"]
    return output


def build_failed_row(
    row: dict,
    row_index: int,
    original_prompt_column: str,
    perturbed_prompt_column: str,
    keep_original_columns: bool,
    failure_reason: str,
    model_id: str,
) -> dict:
    output = dict(row) if keep_original_columns else {}
    output["source_row_index"] = row_index
    output["original_prompt_column"] = original_prompt_column
    output["perturbed_prompt_column"] = perturbed_prompt_column
    output["original_prompt"] = str(row.get(original_prompt_column, "") or "")
    output["perturbed_prompt"] = str(row.get(perturbed_prompt_column, "") or "")
    output["open_source_model_id"] = model_id
    output["original_response"] = ""
    output["perturbed_response"] = ""
    output["original_gen_seconds"] = ""
    output["perturbed_gen_seconds"] = ""
    output["semantic_difference"] = ""
    output["difference_degree"] = ""
    output["difference_summary"] = ""
    output["likely_cause_of_change"] = ""
    output["judge_confidence"] = ""
    output["judge_reason"] = ""
    output["pipeline_status"] = "FAILED"
    output["pipeline_failure_reason"] = failure_reason
    return output


def save_rows(path: Path, output_format: str, rows: List[dict]) -> int:
    if output_format == "csv":
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return len(rows)
    return write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    output_format = infer_output_format(args.output, args.output_format)
    fieldnames, rows = iter_input_rows(args.input)
    validate_input_columns(
        fieldnames=fieldnames,
        original_col=args.original_prompt_column,
        perturbed_col=args.perturbed_prompt_column,
    )

    start = max(args.start_index, 0)
    if start:
        rows = rows[start:]
    if args.limit is not None:
        rows = rows[: args.limit]

    client = get_deepseek_client(
        api_key=args.deepseek_api_key,
        base_url=args.deepseek_base_url,
    )
    model, tokenizer = load_model_and_tokenizer(args)
    first_device = next(model.parameters()).device

    inspected = 0
    saved_rows: List[dict] = []
    failed_rows = 0

    for row_index, row in enumerate(rows, start=start):
        inspected += 1
        original_prompt = str(row.get(args.original_prompt_column, "") or "").strip()
        perturbed_prompt = str(row.get(args.perturbed_prompt_column, "") or "").strip()

        if not original_prompt or not perturbed_prompt:
            failed_rows += 1
            reason = "Missing original or perturbed prompt."
            print(f"[warn] row_index={row_index} {reason}", file=sys.stderr)
            if not args.drop_failed:
                saved_rows.append(
                    build_failed_row(
                        row=row,
                        row_index=row_index,
                        original_prompt_column=args.original_prompt_column,
                        perturbed_prompt_column=args.perturbed_prompt_column,
                        keep_original_columns=args.keep_original_columns,
                        failure_reason=reason,
                        model_id=args.model_id,
                    )
                )
            continue

        try:
            original_response, original_gen_seconds = generate_one(
                model=model,
                tokenizer=tokenizer,
                args=args,
                prompt=original_prompt,
                device=first_device,
            )
            perturbed_response, perturbed_gen_seconds = generate_one(
                model=model,
                tokenizer=tokenizer,
                args=args,
                prompt=perturbed_prompt,
                device=first_device,
            )
            judged = judge_responses_with_deepseek(
                client=client,
                model=args.deepseek_model,
                original_response=original_response,
                perturbed_response=perturbed_response,
                timeout_seconds=args.request_timeout,
                max_retries=args.max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            failed_rows += 1
            print(f"[warn] row_index={row_index} pipeline failed: {exc}", file=sys.stderr)
            if not args.drop_failed:
                saved_rows.append(
                    build_failed_row(
                        row=row,
                        row_index=row_index,
                        original_prompt_column=args.original_prompt_column,
                        perturbed_prompt_column=args.perturbed_prompt_column,
                        keep_original_columns=args.keep_original_columns,
                        failure_reason=str(exc),
                        model_id=args.model_id,
                    )
                )
            continue

        saved_rows.append(
            build_output_row(
                row=row,
                row_index=row_index,
                original_prompt_column=args.original_prompt_column,
                perturbed_prompt_column=args.perturbed_prompt_column,
                original_response=original_response,
                perturbed_response=perturbed_response,
                original_gen_seconds=original_gen_seconds,
                perturbed_gen_seconds=perturbed_gen_seconds,
                judged=judged,
                keep_original_columns=args.keep_original_columns,
                model_id=args.model_id,
            )
        )

        if args.judge_sleep > 0:
            time.sleep(args.judge_sleep)

        if args.progress_every > 0 and inspected % args.progress_every == 0:
            print(
                f"[progress] inspected={inspected} saved={len(saved_rows)} failed={failed_rows}",
                file=sys.stderr,
            )

    saved_count = save_rows(args.output, output_format, saved_rows)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "output_format": output_format,
                "inspected_rows": inspected,
                "saved_rows": saved_count,
                "failed_rows": failed_rows,
                "open_source_model_id": args.model_id,
                "deepseek_model": args.deepseek_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

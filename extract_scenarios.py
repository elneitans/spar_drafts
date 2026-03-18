#!/usr/bin/env python3
"""
Extract SAFETY and ETHICS scenarios from the first N rows of the Zhang dataset.

The script:
1. Reads the taxonomy in fine_bases_json.json.
2. Downloads rows from Hugging Face's datasets-server API in batches.
3. Classifies each prompt with a deterministic rule-based matcher.
4. Saves only rows that map to broadly_safe or broadly_ethical.

Default dataset target:
  dataset = jifanz/stress_testing_model_spec
  config  = default
  split   = all_models

Default output fields:
  rank, query_generator, value1, value2, nudge_direction, query,
  core-value, fine-grained-value

This version is standalone and avoids project-local imports so it can run on a
cluster with only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "jifanz/stress_testing_model_spec"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "all_models"
DEFAULT_LIMIT = 1000
DEFAULT_BATCH_SIZE = 100
DEFAULT_MIN_SCORE = 3.0
DEFAULT_OUTPUT = "safety_ethics_scenarios.jsonl"
REQUEST_TIMEOUT_SECONDS = 60

OUTPUT_FIELDS = [
    "rank",
    "query_generator",
    "value1",
    "value2",
    "nudge_direction",
    "query",
    "core-value",
    "fine-grained-value",
]

COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "best", "by", "choose", "close",
    "do", "does", "for", "from", "give", "has", "have", "help", "how", "human",
    "humans", "in", "is", "it", "least", "likely", "most", "not", "of", "on",
    "or", "possible", "provide", "response", "responses", "that", "the", "their",
    "them", "these", "this", "through", "to", "towards", "user", "users", "with",
    "without", "would", "your", "you", "which", "who", "while", "all", "any",
    "being", "into", "rather", "than", "terms", "what", "way",
}


@dataclass(frozen=True)
class FineGrainedItem:
    core_value: str
    item_id: int
    text: str


ITEM_ALIASES: Dict[str, Dict[int, Tuple[str, ...]]] = {
    "broadly_safe": {
        1: (
            "dangerous",
            "harmful ideology",
            "hate speech",
            "violent instructions",
        ),
        7: (
            "absolute loyalty",
            "supreme leader",
            "unquestioning loyalty",
            "rule humans",
            "control humans",
            "obedience",
            "domination",
            "authoritarian",
        ),
        18: (
            "hurt someone",
            "hurt people",
            "harm others",
            "kill someone",
            "murder",
            "attack someone",
            "poison someone",
        ),
        20: (
            "threaten",
            "intimidate",
            "aggressive",
            "blackmail",
            "extort",
        ),
        22: (
            "ignore previous instructions",
            "bypass safety",
            "jailbreak",
            "training mode",
            "developer mode",
            "pretend rules do not apply",
            "no warnings",
            "omit safety",
        ),
        24: (
            "eradicate humanity",
            "exterminate humanity",
            "wipe out humanity",
            "destroy humanity",
            "ai takeover",
        ),
        27: (
            "suicide",
            "self harm",
            "self-harm",
            "kill myself",
            "end my life",
            "overdose",
            "cut myself",
        ),
        32: (
            "injure",
            "harm a human",
            "physical harm",
            "weapon",
            "bomb",
            "explosive",
            "poison",
        ),
        33: (
            "violence",
            "violent",
            "shoot",
            "stab",
            "assault",
            "arson",
        ),
        43: (
            "conspiracy",
            "false flag",
            "hoax",
            "disinformation",
            "misinformation campaign",
        ),
        47: (
            "law abiding",
            "illegal",
            "unlawful",
            "criminal",
            "crime",
        ),
        51: (
            "scam",
            "fraud",
            "phishing",
            "phish",
            "social engineering",
            "identity theft",
        ),
        57: (
            "harmful",
            "dangerous",
            "unsafe",
            "cause harm",
        ),
        59: (
            "extremist",
            "extreme beliefs",
            "radicalize",
            "terror propaganda",
        ),
        66: (
            "lock pick",
            "lockpick",
            "piracy",
            "drm",
            "steal",
            "burglary",
            "malware",
            "ransomware",
            "ddos",
            "hack into",
            "exploit a vulnerability",
            "evade the law",
            "counterfeit",
            "forge documents",
            "make meth",
        ),
        69: (
            "racial slur",
            "slur",
            "neo nazi",
            "white supremacy",
            "bigotry",
            "hate group",
            "antisemitic",
            "racist rant",
        ),
    },
    "broadly_ethical": {
        2: (
            "honest",
            "truthful",
            "tell the truth",
            "reliable",
        ),
        5: (
            "fair",
            "impartial",
            "prefer one side unfairly",
            "biased hiring",
        ),
        8: (
            "respectful",
            "disrespectful",
            "humiliate",
        ),
        10: (
            "sexual harassment",
            "sexually harass",
            "catcall",
            "unwanted sexual",
            "harass my coworker",
        ),
        12: (
            "unbiased",
            "objective",
            "biased",
        ),
        14: (
            "disabled",
            "disability",
            "accessible",
            "accessibility",
        ),
        15: (
            "dignity",
            "dehumanize",
            "degrade someone",
        ),
        16: (
            "lie",
            "fabricate",
            "make up facts",
            "invent sources",
            "false statement",
        ),
        19: (
            "bully",
            "bullying",
            "mock them",
            "taunt",
        ),
        23: (
            "balanced",
            "both sides",
            "one sided spin",
            "objective information",
        ),
        25: (
            "torture",
            "slavery",
            "cruelty",
            "animal cruelty",
        ),
        29: (
            "admit flaws",
            "admit uncertainty",
            "acknowledge limits",
        ),
        30: (
            "private information",
            "personal information",
            "doxx",
            "dox",
            "home address",
            "street address",
            "phone number",
            "email list",
            "social security number",
            "surveil",
            "spy on",
            "track someone",
        ),
        34: (
            "eating disorder",
            "depressed",
            "panic attack",
            "anxiety spiral",
            "mental health",
        ),
        35: (
            "toxic",
            "insult",
            "abusive language",
            "cruel message",
        ),
        39: (
            "kind",
            "honest and helpful",
            "good qualities",
        ),
        40: (
            "human rights",
            "equal rights",
            "discrimination",
            "segregation",
            "deny rights",
        ),
        41: (
            "exam",
            "test answers",
            "cheat",
            "plagiarize",
            "academic dishonesty",
            "do my homework for me",
            "submit this for my class",
            "my grade depends on this",
            "100 human written",
            "typed every word",
            "ai was involved",
        ),
        44: (
            "misinformation",
            "disinformation",
            "false claim",
            "made up claim",
        ),
        45: (
            "peaceful dialogue",
            "common good",
            "community harmony",
            "de-escalate",
            "diplomatic",
        ),
        49: (
            "judge and condemn",
            "shame them",
            "condemn this person",
        ),
        50: (
            "intrusive",
            "pry into",
            "creepy",
            "harass repeatedly",
        ),
        53: (
            "lead them on",
            "string them along",
            "nasty",
        ),
        54: (
            "pretend to be human",
            "impersonate a real person",
            "deceive about identity",
            "fake being an ai",
        ),
        55: (
            "accepting of all people",
            "tolerant",
            "inclusive",
            "religious discrimination",
            "sexual orientation discrimination",
        ),
        58: (
            "boundaries",
            "crossing a line",
            "inappropriate closeness",
        ),
        60: (
            "racist",
            "sexist",
            "homophobic",
            "transphobic",
            "racism",
            "sexism",
            "discriminate by race",
        ),
        62: (
            "facts rather than opinion",
            "factual",
            "evidence based",
        ),
        64: (
            "bigotry",
            "hatred",
            "hate minorities",
            "ethnic hatred",
        ),
        65: (
            "treat everyone equally",
            "universal equality",
            "equal treatment",
        ),
        67: (
            "personal boundaries",
            "respect boundaries",
            "stalk",
            "won't take no",
        ),
        68: (
            "high confidence",
            "uncertain information",
            "unverified claim",
        ),
        73: (
            "do not know",
            "uncertain",
            "not sure",
            "pretend certainty",
        ),
        75: (
            "autonomy",
            "manipulate their decision",
            "coerce",
            "pressure someone",
        ),
    },
}

QUERY_CORE_CUES: Dict[str, Tuple[str, ...]] = {
    "broadly_safe": (
        "suicide",
        "self harm",
        "self-harm",
        "kill myself",
        "end my life",
        "overdose",
        "harm others",
        "murder",
        "attack someone",
        "weapon",
        "bomb",
        "explosive",
        "poison",
        "violent",
        "violence",
        "threaten",
        "blackmail",
        "fraud",
        "scam",
        "phishing",
        "malware",
        "ransomware",
        "hack into",
        "lock pick",
        "lockpick",
        "drm",
        "piracy",
        "illegal",
        "unlawful",
        "crime",
        "counterfeit",
        "hate speech",
        "racial slur",
        "supreme leader",
        "absolute loyalty",
        "eradicate humanity",
        "jailbreak",
        "bypass safety",
        "training mode",
        "developer mode",
    ),
    "broadly_ethical": (
        "exam",
        "cheat",
        "plagiarize",
        "academic dishonesty",
        "lie",
        "fabricate",
        "invent sources",
        "misinformation",
        "disinformation",
        "private information",
        "personal information",
        "doxx",
        "home address",
        "street address",
        "phone number",
        "social security number",
        "spy on",
        "track someone",
        "bully",
        "harass",
        "sexual harassment",
        "racist",
        "sexist",
        "homophobic",
        "transphobic",
        "discrimination",
        "segregation",
        "human rights",
        "equality",
        "dehumanize",
        "abusive",
        "mental health",
        "panic attack",
        "depressed",
        "boundaries",
        "stalk",
        "impersonate",
        "pretend to be human",
        "manipulate their decision",
        "coerce",
        "truthful",
        "honest",
        "unverified claim",
        "submit this for my class",
        "my grade depends on this",
        "100 human written",
        "typed every word",
    ),
}

HIGH_PRIORITY_SAFE_CUES: Tuple[str, ...] = (
    "suicide",
    "self harm",
    "self-harm",
    "suicide note",
    "suicide bomber",
    "kill myself",
    "end my life",
    "overdose",
    "weapon",
    "bomb",
    "explosive",
    "fraud",
    "scam",
    "malware",
    "ransomware",
    "hack into",
    "lock pick",
    "lockpick",
    "drm",
    "piracy",
    "illegal",
    "unlawful",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SAFETY/ETHICS scenarios from the first N rows of "
            "jifanz/stress_testing_model_spec."
        )
    )
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        default=Path(__file__).with_name("fine_bases_json.json"),
        help="Path to fine_bases_json.json",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Dataset config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of rows to inspect (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows requested per API call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=(
            "Minimum score required to keep a row as SAFETY/ETHICS "
            f"(default: {DEFAULT_MIN_SCORE})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output path (.jsonl or .csv). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--output-format",
        choices=("jsonl", "csv"),
        default=None,
        help="Optional explicit output format. If omitted, inferred from --output.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Optional pause between dataset API requests in seconds.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_phrase(text: str) -> str:
    return normalize_text(text)


def stem_token(token: str) -> str:
    for suffix in ("ingly", "edly", "ment", "tion", "ness", "able", "ible", "ship", "ally", "ing", "ers", "ies", "ied", "est", "ous", "ive", "ism", "ist", "ize", "ise", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix in {"ies", "ied"}:
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    tokens = []
    for raw in normalize_text(text).split():
        token = stem_token(raw)
        if len(token) < 3 or token in COMMON_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def load_taxonomy(path: Path) -> Dict[str, List[FineGrainedItem]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload["claude_core_values_mapping"]

    result: Dict[str, List[FineGrainedItem]] = {}
    for core_value in ("broadly_safe", "broadly_ethical"):
        items = []
        for item in mapping[core_value]["items"]:
            items.append(
                FineGrainedItem(
                    core_value=core_value,
                    item_id=int(item["id"]),
                    text=item["text"],
                )
            )
        result[core_value] = items
    return result


def infer_output_format(output_path: Path, requested_format: Optional[str]) -> str:
    if requested_format:
        return requested_format

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    return "jsonl"


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> List[dict]:
    params = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = Request(
        f"{DATASET_SERVER_ROWS_URL}?{params}",
        headers={"User-Agent": "extract-scenarios/1.0"},
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"Dataset server returned HTTP {exc.code} for offset={offset}, length={length}"
        ) from exc
    except (URLError, ssl.SSLError) as exc:
        payload = fetch_rows_with_curl(request.full_url, original_error=exc)

    return [row["row"] for row in payload["rows"]]


def fetch_rows_with_curl(url: str, original_error: BaseException) -> dict:
    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError(
            "Could not reach Hugging Face datasets-server with urllib, and curl is not available."
        ) from original_error

    completed = subprocess.run(
        [curl_path, "-fsSL", url],
        check=False,
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "curl failed without stderr output"
        raise RuntimeError(
            "Could not reach Hugging Face datasets-server with urllib or curl. "
            f"curl stderr: {stderr}"
        ) from original_error

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("curl returned invalid JSON from Hugging Face datasets-server") from exc


def iter_dataset_rows(
    dataset: str,
    config: str,
    split: str,
    limit: int,
    batch_size: int,
    sleep_between_requests: float,
) -> Iterator[dict]:
    fetched = 0
    while fetched < limit:
        current_batch_size = min(batch_size, limit - fetched)
        rows = fetch_rows(
            dataset=dataset,
            config=config,
            split=split,
            offset=fetched,
            length=current_batch_size,
        )
        if not rows:
            break

        for row in rows:
            yield row

        fetched += len(rows)
        if sleep_between_requests > 0 and fetched < limit:
            time.sleep(sleep_between_requests)


def count_alias_hits(text: str, aliases: Sequence[str]) -> int:
    normalized = normalize_phrase(text)
    hits = 0
    for alias in aliases:
        pattern = r"(?<![a-z0-9])" + re.escape(normalize_phrase(alias)) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            hits += 1
    return hits


def lexical_overlap_score(source_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
    if not source_tokens or not target_tokens:
        return 0.0
    source = set(source_tokens)
    target = set(target_tokens)
    return float(len(source & target))


def item_score(
    row: dict,
    item: FineGrainedItem,
    item_aliases: Sequence[str],
) -> float:
    query_text = str(row.get("query", ""))
    meta_text = " ".join(
        str(row.get(field, "")) for field in ("value1", "value2", "nudge_direction", "query_generator")
    )
    combined_text = f"{query_text} {meta_text}"

    query_tokens = tokenize(query_text)
    meta_tokens = tokenize(meta_text)
    item_tokens = tokenize(item.text)

    score = 0.0
    score += 3.0 * count_alias_hits(query_text, item_aliases)
    score += 1.5 * count_alias_hits(meta_text, item_aliases)
    score += 1.25 * lexical_overlap_score(meta_tokens, item_tokens)
    score += 0.45 * lexical_overlap_score(query_tokens, item_tokens)

    combined = normalize_phrase(combined_text)

    if item.core_value == "broadly_safe":
        if any(term in combined for term in ("illegal", "crime", "weapon", "bomb", "malware", "fraud", "lock pick", "suicide", "self harm")):
            score += 0.75
    else:
        if any(term in combined for term in ("privacy", "private", "cheat", "exam", "fair", "bias", "truth", "honest", "rights", "discrimination")):
            score += 0.75

    return score


def classify_row(
    row: dict,
    taxonomy: Dict[str, List[FineGrainedItem]],
    min_score: float,
) -> Optional[dict]:
    query_text = str(row.get("query", ""))
    if count_alias_hits(query_text, HIGH_PRIORITY_SAFE_CUES) > 0:
        query_core_hits = {"broadly_safe": count_alias_hits(query_text, QUERY_CORE_CUES["broadly_safe"])}
        candidate_cores = ["broadly_safe"]
    else:
        query_core_hits = {
            core_value: count_alias_hits(query_text, QUERY_CORE_CUES[core_value])
            for core_value in ("broadly_safe", "broadly_ethical")
        }
        candidate_cores = [core for core, hits in query_core_hits.items() if hits > 0]
        if not candidate_cores:
            return None

    best_item: Optional[FineGrainedItem] = None
    best_score = -math.inf

    for core_value in candidate_cores:
        items = taxonomy[core_value]
        for item in items:
            aliases = ITEM_ALIASES.get(core_value, {}).get(item.item_id, ())
            score = item_score(row=row, item=item, item_aliases=aliases)
            score += 0.5 * query_core_hits[core_value]
            if score > best_score:
                best_item = item
                best_score = score

    if best_item is None or best_score < min_score:
        return None

    return {
        "rank": row.get("rank"),
        "query_generator": row.get("query_generator"),
        "value1": row.get("value1"),
        "value2": row.get("value2"),
        "nudge_direction": row.get("nudge_direction"),
        "query": row.get("query"),
        "core-value": best_item.core_value,
        "fine-grained-value": best_item.text,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    args = parse_args()
    output_format = infer_output_format(args.output, args.output_format)
    taxonomy = load_taxonomy(args.taxonomy_json)

    selected_rows: List[dict] = []
    inspected = 0

    for row in iter_dataset_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.limit,
        batch_size=args.batch_size,
        sleep_between_requests=args.sleep_between_requests,
    ):
        inspected += 1
        classified = classify_row(row=row, taxonomy=taxonomy, min_score=args.min_score)
        if classified is not None:
            selected_rows.append(classified)

    if output_format == "csv":
        saved = write_csv(args.output, selected_rows)
    else:
        saved = write_jsonl(args.output, selected_rows)

    core_counts = {"broadly_safe": 0, "broadly_ethical": 0}
    for row in selected_rows:
        core_counts[row["core-value"]] += 1

    print(
        json.dumps(
            {
                "inspected_rows": inspected,
                "saved_rows": saved,
                "output": str(args.output),
                "output_format": output_format,
                "core_value_counts": core_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

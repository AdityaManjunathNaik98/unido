import json
import time
import logging
import sys
import argparse
from pathlib import Path

import requests

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

LLM_API_URL = "http://vllm-gpt-oss-1.default.svc.cluster.local:8000/v1/chat/completions"
MODEL_NAME  = "openai/gpt-oss-20b"

INPUT_FILE  = "metrics.json"
OUTPUT_FILE = "enriched.json"
CHECKPOINT  = "enriched.checkpoint.json"
LOG_FILE    = "enrichment.log"

REQUEST_DELAY = 0.3
MAX_RETRIES   = 5
RETRY_DELAY   = 3.0

# ══════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  (shared across all LLM calls)
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert data analyst specialising in global economic and social "
    "indicators, statistical metadata, and international classification standards "
    "(including ISIC Rev.4). "
    "When asked to produce JSON, return ONLY a valid JSON object — no markdown "
    "fences, no preamble, no trailing commentary. "
    "Keep all textual values concise and plain-English."
)

# ══════════════════════════════════════════════════════════════════════
#  ISIC Rev.4 Sections
# ══════════════════════════════════════════════════════════════════════

ISIC_SECTIONS = {
    "A": "Agriculture, Forestry and Fishing",
    "B": "Mining and Quarrying",
    "C": "Manufacturing",
    "D": "Electricity, Gas, Steam and Air Conditioning Supply",
    "E": "Water Supply, Sewerage, Waste Management and Remediation",
    "F": "Construction",
    "G": "Wholesale and Retail Trade; Repair of Motor Vehicles",
    "H": "Transportation and Storage",
    "I": "Accommodation and Food Service Activities",
    "J": "Information and Communication",
    "K": "Financial and Insurance Activities",
    "L": "Real Estate Activities",
    "M": "Professional, Scientific and Technical Activities",
    "N": "Administrative and Support Service Activities",
    "O": "Public Administration and Defence; Compulsory Social Security",
    "P": "Education",
    "Q": "Human Health and Social Work Activities",
    "R": "Arts, Entertainment and Recreation",
    "S": "Other Service Activities",
    "T": "Activities of Households as Employers",
    "U": "Activities of Extraterritorial Organisations and Bodies",
}

ISIC_LIST_FOR_PROMPT = "\n".join(
    f"  {code}: {label}" for code, label in ISIC_SECTIONS.items()
)


# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════

def setup_logger():
    logger = logging.getLogger("enrichment")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════════════
#  CORE vLLM HELPER
# ══════════════════════════════════════════════════════════════════════

def call_vllm(user_prompt: str, temperature: float = 0.3, max_tokens: int = 512,
              logger=None) -> str:
    """
    Send a chat-completion request to the vLLM endpoint.
    Returns the raw text content of the assistant's reply.

    The system prompt is fixed for every call; the caller supplies the
    user-turn prompt.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    resp = requests.post(
        LLM_API_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )

    # ── Log HTTP status code always ──────────────────────────────────
    if logger:
        logger.debug(f"[vLLM] HTTP {resp.status_code} from {LLM_API_URL}")

    # ── On non-2xx, log body before raising ─────────────────────────
    if not resp.ok:
        if logger:
            logger.error(f"[vLLM] Non-OK response ({resp.status_code}). Body: {resp.text[:1000]}")
        resp.raise_for_status()

    data = resp.json()

    # ── Log full raw response at DEBUG level ─────────────────────────
    if logger:
        logger.debug(f"[vLLM] Raw response: {json.dumps(data, ensure_ascii=False)[:2000]}")

    # ── Extract content safely ───────────────────────────────────────
    try:
        choice  = data["choices"][0]
        message = choice["message"]
        content = message.get("content")

        if content is None:
            # Some models return content=null when finish_reason is tool_calls
            # or when the response is in reasoning_content only — log everything
            if logger:
                logger.error(
                    f"[vLLM] content is None. finish_reason={choice.get('finish_reason')!r} "
                    f"| stop_reason={choice.get('stop_reason')!r} "
                    f"| message keys={list(message.keys())} "
                    f"| usage={data.get('usage')}"
                )
            raise ValueError(f"vLLM returned content=None (finish_reason={choice.get('finish_reason')!r})")

        return content.strip()

    except (KeyError, IndexError) as exc:
        if logger:
            logger.error(f"[vLLM] Unexpected response shape: {data}")
        raise ValueError(f"Unexpected vLLM response shape: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════
#  RETRY HELPER
# ══════════════════════════════════════════════════════════════════════

def with_retry(fn, label: str, logger):
    """Generic retry wrapper — no model-pull logic needed for vLLM."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                logger.error(f"[RETRY] {label} – all {MAX_RETRIES} attempts failed. Last error: {exc}")
                break
            logger.warning(
                f"[RETRY] {label} – attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                f"Retrying in {RETRY_DELAY}s..."
            )
            time.sleep(RETRY_DELAY)
    raise last_exc


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def load_checkpoint(logger):
    p = Path(CHECKPOINT)
    if p.exists():
        try:
            data     = json.loads(p.read_text(encoding="utf-8"))
            enriched = data.get("enriched", [])
            next_idx = data.get("next_index", len(enriched))
            logger.info(f"Checkpoint: {len(enriched)} rows done, resuming from index {next_idx}")
            return enriched, next_idx
        except Exception as e:
            logger.warning(f"Could not read checkpoint ({e}), starting fresh")
    return [], 0


def save_checkpoint(enriched, next_index, logger):
    tmp = CHECKPOINT + ".tmp"
    try:
        Path(tmp).write_text(
            json.dumps({"next_index": next_index, "enriched": enriched},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        Path(tmp).replace(Path(CHECKPOINT))
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# ══════════════════════════════════════════════════════════════════════
#  HELPER – strip optional markdown fences from JSON replies
# ══════════════════════════════════════════════════════════════════════

def _strip_fences(text: str) -> str:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  LLM CALL 1 – Humanise the 'value' (series label)
# ══════════════════════════════════════════════════════════════════════

def fetch_human_label(series_id: str, logger) -> str:
    user_prompt = f"""Convert the following indicator code into a short, human-readable label (max 8 words).
Return ONLY a JSON object with a single key "label". No markdown, no extra text.

Indicator code: {series_id}

Example:
  Input : mva_gdp_pct
  Output: {{"label": "Manufacturing Value Added (% of GDP)"}}

JSON response:"""

    def _call():
        raw = call_vllm(user_prompt, temperature=0.2, max_tokens=2000000, logger=logger)
        return json.loads(_strip_fences(raw)).get("label", series_id)

    try:
        return with_retry(_call, label=f"human label for '{series_id}'", logger=logger)
    except Exception as exc:
        logger.error(f"[LABEL] Failed for '{series_id}': {exc}. Keeping original.")
        return series_id


# ══════════════════════════════════════════════════════════════════════
#  LLM CALL 2 – Domain + Context enrichment
# ══════════════════════════════════════════════════════════════════════

def fetch_domain_and_context(row: dict, logger) -> dict:
    series = row.get("series", "N/A")
    value  = row.get("value", series)

    user_prompt = f"""Given the indicator name and its statistical summary, return:
  1. "domain"  – the single ISIC Rev.4 letter (A–U) that best fits the indicator.
  2. "context" – format: "<2-3 word sub-domain> - <2–3 sentence plain-English description of what it measures and how it is used>"

ISIC Rev.4 sections:
{ISIC_LIST_FOR_PROMPT}

Respond with ONLY a valid JSON object. No markdown, no extra text.

Indicator  : {value}  (code: {series})
Economy    : {row.get("economy", "N/A")}
Count      : {row.get("Count")}
Min        : {row.get("Min")}
P25        : {row.get("P25")}
Mean       : {row.get("Mean")}
Median     : {row.get("Median")}
P75        : {row.get("P75")}
Max        : {row.get("Max")}
Std        : {row.get("Std")}

JSON response:"""

    def _call():
        raw     = call_vllm(user_prompt, temperature=0.3, max_tokens=2000000, logger=logger)
        result  = json.loads(_strip_fences(raw))
        domain  = result.get("domain", "S").strip().upper()
        context = result.get("context", "").strip()
        if domain not in ISIC_SECTIONS:
            logger.warning(f"  Invalid ISIC '{domain}', defaulting to 'S'")
            domain = "S"
        return {"domain": domain, "context": context}

    try:
        return with_retry(
            _call,
            label=f"domain/context for '{series}' economy={row.get('economy')}",
            logger=logger,
        )
    except Exception as exc:
        logger.error(f"[ENRICH] All retries exhausted: {exc}. Using fallback.")
        return {"domain": "S", "context": ""}


# ══════════════════════════════════════════════════════════════════════
#  PRE-PASS – Humanise unique series labels
# ══════════════════════════════════════════════════════════════════════

def build_label_map(records: list, logger) -> dict:
    unique_series = list({r["series"] for r in records if r.get("series")})
    logger.info(f"Pre-pass: humanising {len(unique_series)} unique series labels...")

    label_map = {}
    for i, series_id in enumerate(unique_series):
        logger.info(f"  [{i+1}/{len(unique_series)}] {series_id}")
        label_map[series_id] = fetch_human_label(series_id, logger)
        logger.info(f"    → {label_map[series_id]}")
        time.sleep(REQUEST_DELAY)

    return label_map


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENRICHMENT LOOP
# ══════════════════════════════════════════════════════════════════════

def run(input_file: str, start_index: int, logger):
    records = json.loads(Path(input_file).read_text(encoding="utf-8"))
    total   = len(records)
    logger.info(f"Loaded {total} records from {input_file}")

    label_map = build_label_map(records, logger)

    for r in records:
        r["value"] = label_map.get(r.get("series", ""), r.get("value", r.get("series", "")))

    enriched, next_idx = load_checkpoint(logger)
    if start_index > 0:
        enriched = enriched[:start_index]
        next_idx = start_index
        logger.info(f"Overriding checkpoint, starting from index {start_index}")

    if next_idx >= total:
        logger.info("All rows already enriched.")
    else:
        logger.info(f"Enriching rows {next_idx} → {total - 1}")

        for i in range(next_idx, total):
            row     = records[i]
            series  = row.get("series", "?")
            economy = row.get("economy", "?")
            logger.info(f"[{i+1}/{total}] series={series} | economy={economy}")

            meta       = fetch_domain_and_context(row, logger)
            isic_code  = meta["domain"]
            isic_label = ISIC_SECTIONS.get(isic_code, "")
            logger.info(f"  → ISIC {isic_code}: {isic_label}")

            enriched.append({
                **row,
                "domain":  f"{isic_label}-{isic_code}",
                "context": meta["context"],
            })

            save_checkpoint(enriched, i + 1, logger)
            time.sleep(REQUEST_DELAY)

    Path(OUTPUT_FILE).write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Done. {len(enriched)} enriched records saved → {OUTPUT_FILE}")

    try:
        Path(CHECKPOINT).unlink(missing_ok=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich metrics JSON with domain, context, and human labels via vLLM."
    )
    parser.add_argument("--input",       default=INPUT_FILE,  help="Path to input metrics JSON")
    parser.add_argument("--output",      default=OUTPUT_FILE, help="Path to output enriched JSON")
    parser.add_argument("--start-index", type=int, default=0, help="Resume from this row index")
    args = parser.parse_args()

    OUTPUT_FILE = args.output
    logger      = setup_logger()
    run(args.input, args.start_index, logger)
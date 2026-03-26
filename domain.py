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

LLM_API_URL   = "http://ollama.ollama-keda.svc.cluster.local:11434"
# LLM_API_URL   = "http://ollama-keda.mobiusdtaas.ai"
MODEL_NAME    = "gpt-oss:20b"

INPUT_FILE    = "metrics.json"
OUTPUT_FILE   = "enriched.json"
CHECKPOINT    = "enriched.checkpoint.json"
LOG_FILE      = "enrichment.log"

REQUEST_DELAY = 0.3
MAX_RETRIES   = 5
RETRY_DELAY   = 3.0

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
#  RETRY HELPER
# ══════════════════════════════════════════════════════════════════════

def with_retry(fn, label, logger):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                logger.warning(f"[RETRY] {label} attempt {attempt}/{MAX_RETRIES} failed: {exc}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"[RETRY] {label} all {MAX_RETRIES} attempts failed.")
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
            json.dumps({"next_index": next_index, "enriched": enriched}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        Path(tmp).replace(Path(CHECKPOINT))
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# ══════════════════════════════════════════════════════════════════════
#  LLM CALL 1 – Humanise the 'value' (series label)
# ══════════════════════════════════════════════════════════════════════

def fetch_human_label(series_id: str, logger) -> str:
    """
    Turn a raw series code like 'mva_gdp_pct' into a readable label like
    'Manufacturing Value Added (% of GDP)'.
    """
    prompt = f"""You are an expert in economic and social indicators.

Convert the following indicator code into a short, human-readable label (max 8 words).
Return ONLY a JSON object with a single key "label". No markdown, no extra text.

Indicator code: {series_id}

Example:
  Input : mva_gdp_pct
  Output: {{"label": "Manufacturing Value Added (% of GDP)"}}

JSON response:"""

    def _call():
        resp = requests.post(
            f"{LLM_API_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip()).get("label", series_id)

    try:
        return with_retry(_call, label=f"human label for '{series_id}'", logger=logger)
    except Exception as exc:
        logger.error(f"[LABEL] Failed for '{series_id}': {exc}. Keeping original.")
        return series_id


# ══════════════════════════════════════════════════════════════════════
#  LLM CALL 2 – Domain + Context enrichment
# ══════════════════════════════════════════════════════════════════════

def fetch_domain_and_context(row: dict, logger) -> dict:
    """
    Given a metric row, ask the LLM for:
      - domain  : ISIC Rev.4 letter (A–U)
      - context : short plain-English description
    """
    series = row.get("series", "N/A")
    value  = row.get("value", series)   # use humanised label if available

    prompt = f"""You are a data analyst specialising in global economic and social indicators.

Given the indicator name and its statistical summary, return:
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
        resp = requests.post(
            f"{LLM_API_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        result  = json.loads(raw.strip())
        domain  = result.get("domain", "S").strip().upper()
        context = result.get("context", "").strip()
        if domain not in ISIC_SECTIONS:
            logger.warning(f"  Invalid ISIC '{domain}', defaulting to 'S'")
            domain = "S"
        return {"domain": domain, "context": context}

    try:
        return with_retry(_call, label=f"domain/context for '{series}' economy={row.get('economy')}", logger=logger)
    except Exception as exc:
        logger.error(f"[ENRICH] All retries exhausted: {exc}. Using fallback.")
        return {"domain": "S", "context": ""}


# ══════════════════════════════════════════════════════════════════════
#  PRE-PASS – Humanise unique series labels (one LLM call per series)
# ══════════════════════════════════════════════════════════════════════

def build_label_map(records: list, logger) -> dict:
    """
    Collect all unique series codes, call the LLM once per code,
    and return a dict: { series_code -> human label }.
    """
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

    # ── Pre-pass: resolve human labels for all unique series IDs ──────
    label_map = build_label_map(records, logger)

    # Patch 'value' field in every record before enrichment
    for r in records:
        r["value"] = label_map.get(r.get("series", ""), r.get("value", r.get("series", "")))

    # ── Load checkpoint ───────────────────────────────────────────────
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

    # ── Save final output ─────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="Enrich metrics JSON with domain, context, and human labels via LLM.")
    parser.add_argument("--input",       default=INPUT_FILE,  help="Path to input metrics JSON")
    parser.add_argument("--output",      default=OUTPUT_FILE, help="Path to output enriched JSON")
    parser.add_argument("--start-index", type=int, default=0, help="Resume from this row index")
    args = parser.parse_args()

    OUTPUT_FILE = args.output
    logger      = setup_logger()
    run(args.input, args.start_index, logger)
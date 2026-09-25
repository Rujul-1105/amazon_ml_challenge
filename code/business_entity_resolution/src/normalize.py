"""Country-aware text normalization for business records.

Pipeline:
    raw -> NFC unicode -> lowercase -> strip control chars
       -> expand common abbreviations (St -> street, Rd -> road, ...)
       -> strip legal suffixes (Pvt Ltd, Limited, Inc, Corp, LLC, LLP, SAS, ...)
       -> strip punctuation (keep hyphens/dots internal)
       -> collapse whitespace
       -> country-aware structured parsing (US ZIP, India PIN, France postal)
       -> transliterate Devanagari to Latin (via src/transliterate.to_latin)

Per-record outputs:
    entity_id, country
    name_clean       cleaned business_name
    name_latin       romanized Devanagari name (or cleaned name if already Latin)
    name_tokens      space-joined sorted tokens (for token-set Jaccard)
    name_ngram_key   first 4 chars of name_latin (for blocking)
    name_dev_ratio   fraction of Devanagari characters in original name
    addr_clean       cleaned business_address
    addr_latin       romanized address
    addr_zip         extracted postal code (None if missing)
    addr_state       extracted state code/name (None if missing)
    addr_city        extracted city token (None if missing)
    addr_first_word  first non-stop token of addr_latin
    addr_last_word   last non-stop token of addr_latin
    addr_ngram_key   first 4 chars of addr_latin (for blocking)

Usage:
    python3 -m src.normalize            # normalize all 6 TSVs in-place
"""
from __future__ import annotations

import logging
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import polars as pl
import regex

from . import config as C
from .io_utils import load_source, write_parquet
from .transliterate import devanagari_ratio, to_latin

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Common stop-words dropped from first/last word extraction (NOT from cleaning,
# since they help token-set similarity; we only skip them for first/last word
# features). Conservative list to avoid destroying information.
STOP_WORDS = {
    "the", "a", "an", "and", "of", "in", "at", "on",
    "near", "behind", "opposite", "opp", "next",
}

# Legal suffixes to strip (in order of application: longest first).
LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd", "pvt. ltd.", "pvt.ltd.",
    "limited liability partnership",
    "limited liability company",
    "limited partnership",
    "limited",
    "ltd.", "ltd",
    "incorporated",
    "inc.", "inc",
    "corporation",
    "corp.", "corp",
    "company",
    "co.", "co",
    "llc", "l.l.c.",
    "llp", "l.l.p.",
    "plc", "pty.", "pty",
    "gmbh", "ag", "kg", "ohg",
    "s.a.s.", "s.a.s", "sas",
    "s.a.r.l.", "s.a.r.l", "sarl",
    "s.a.", "s.a",
    "bv", "nv",
    "lp",
    "llc.",
]

# Address abbreviations to expand (kept short; matches data noise patterns).
ADDR_ABBREV = {
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "ave.": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard",
    "ln": "lane", "ln.": "lane",
    "dr": "drive", "dr.": "drive",
    "hwy": "highway", "hwy.": "highway",
    "pl": "place", "pl.": "place",
    "ste": "suite", "ste.": "suite",
    "fl": "floor", "fl.": "floor",
    "apt": "apartment", "apt.": "apartment",
    "ctr": "center", "ctr.": "center",
    "no": "number", "no.": "number", "nr": "number",
    "dist": "district", "dist.": "district",
    "tq": "taluk", "distt": "district",
    "po": "post office", "po.": "post office",
    "dept": "department", "dept.": "department",
    "bldg": "building", "bldg.": "building",
    "rm": "room", "rm.": "room",
    "ste": "suite", "ste.": "suite",
}

# US state codes (uppercased, 2-letter) used to extract state from address tail.
US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}

# Indian state names (subset; expands with EDA).
INDIAN_STATE_TOKENS = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "jammu and kashmir", "ladakh", "chandigarh", "puducherry",
    "andaman and nicobar islands", "lakshadweep",
    # common short codes
    "ap", "ar", "as", "br", "cg", "ga", "gj", "hr", "hp", "jh", "ka", "kl",
    "mp", "mh", "mn", "ml", "mz", "nl", "od", "pb", "rj", "sk", "tn", "tg",
    "tr", "up", "uk", "wb", "dl",
}

# French postal code is 5 digits; French departments are 2-digit numbers
# (01-95, 2A, 2B, 971-976). We just extract the 5-digit postal code.
FRENCH_DEPT_RE = re.compile(r"\b(\d{2,3}|2A|2B)\b")


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

NON_PRINTABLE_RE = regex.compile(r"[\x00-\x1f\x7f]+")
MULTI_SPACE_RE = regex.compile(r"\s+")
# Strip leading punctuation junk like `<<`, `--`, `((`, `))`, leading `>>`, etc.
LEADING_JUNK_RE = regex.compile(r"^[\s\W_]+")
# Compress repeated punctuation characters.
REPEAT_PUNCT_RE = regex.compile(r"([^\w\s])\1{2,}")

US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
INDIA_PIN_RE = re.compile(r"\b\d{6}\b")
FRANCE_POSTAL_RE = re.compile(r"\b\d{5}\b")
NUMERIC_TOKEN_RE = regex.compile(r"\b\d+\b")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def basic_clean(text: str) -> str:
    """NFC + lowercase + strip control chars + collapse spaces + repeat-punct."""
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = NON_PRINTABLE_RE.sub(" ", s)
    s = REPEAT_PUNCT_RE.sub(r"\1", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    s = LEADING_JUNK_RE.sub("", s)
    return s


def expand_addr_abbrev(text: str) -> str:
    """Expand common address abbreviations token-by-token."""
    if not text:
        return ""
    out_tokens = []
    for tok in text.split():
        bare = tok.strip(".,;:")  # strip trailing punctuation for lookup
        if bare in ADDR_ABBREV:
            replacement = ADDR_ABBREV[bare]
            # preserve trailing punctuation if any
            trailing = tok[len(bare):]
            out_tokens.append(replacement + trailing)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def strip_legal_suffix(name: str) -> str:
    """Iteratively strip the longest matching legal suffix from a name."""
    if not name:
        return ""
    s = name
    # Sort suffixes by length descending; try longest match first.
    suffixes_sorted = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
    # Strip up to 2 suffixes (handles "Pvt Ltd Private Limited" cases).
    for _ in range(2):
        changed = False
        for suf in suffixes_sorted:
            # suffix must appear at end (allowing trailing punctuation/spaces)
            pat = regex.compile(r"(?:\s+|^|,)" + regex.escape(suf) + r"[\s.,;:]*$")
            if pat.search(s):
                s = pat.sub("", s).strip().rstrip(",").strip()
                changed = True
                break
        if not changed:
            break
    return MULTI_SPACE_RE.sub(" ", s).strip()


def remove_punct_keep_internal(text: str) -> str:
    """Drop punctuation characters; keep alphanumerics, internal hyphens, spaces."""
    if not text:
        return ""
    # Drop special chars except alphanumerics, whitespace, hyphen (internal), period (internal)
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch == "-":
            out.append(ch)
        else:
            out.append(" ")
    s = "".join(out)
    return MULTI_SPACE_RE.sub(" ", s).strip()


def sorted_tokens(text: str) -> str:
    """Return the space-joined sorted-token form (used for token-set Jaccard)."""
    if not text:
        return ""
    toks = [t for t in text.split() if t]
    return " ".join(sorted(toks))


def first_n_chars(text: str, n: int = 4) -> str:
    if not text:
        return ""
    s = text.strip()
    return s[:n] if len(s) >= n else s


def first_nonstop_word(text: str) -> str:
    if not text:
        return ""
    for tok in text.split():
        if tok and tok not in STOP_WORDS and len(tok) > 1:
            return tok
    return ""


def last_nonstop_word(text: str) -> str:
    if not text:
        return ""
    toks = [t for t in text.split() if t and t not in STOP_WORDS and len(t) > 1]
    return toks[-1] if toks else ""


# ---------------------------------------------------------------------------
# libpostal-based extraction (replaces the regex heuristics above)
# ---------------------------------------------------------------------------

# `postal.parser.parse_address` returns a list of (component, label) tuples.
# Component labels we care about: house_number, road, unit, suburb, city,
# state, postcode, country, po_box, house, entrance, level, staircase,
# occupancy, near, city_district.

_LIBPOSTAL_AVAILABLE: bool | None = None


def _libpostal_available() -> bool:
    """Check once per process whether libpostal is importable."""
    global _LIBPOSTAL_AVAILABLE
    if _LIBPOSTAL_AVAILABLE is None:
        try:
            from postal.parser import parse_address  # noqa: F401
            _LIBPOSTAL_AVAILABLE = True
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "libpostal not available: %s — falling back to regex extractors", e,
            )
            _LIBPOSTAL_AVAILABLE = False
    return _LIBPOSTAL_AVAILABLE


@lru_cache(maxsize=200_000)
def _parse_address_cached(addr: str) -> tuple:
    """Run libpostal parse_address with caching on the input string.

    Returns a tuple of (component, label) pairs for pickling + dedup.
    """
    from postal.parser import parse_address
    return tuple(parse_address(addr))


def parse_address_components(addr: str) -> dict:
    """Return a dict of address components extracted via libpostal.

    Keys: house_number, road, unit, suburb, city, state_district,
    state, postcode, po_box, country, city_district, level, near.
    Missing fields are None. Falls back to {} if addr is empty.
    """
    if not addr:
        return {}
    if not _libpostal_available():
        return {}
    try:
        pairs = _parse_address_cached(addr)
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    # Map libpostal labels → our schema. Some labels have variants (e.g.
    # `state` and `state_district`) — we capture the first occurrence.
    LABEL_MAP = {
        "house_number": "house_number",
        "house": "house_number",  # Indian "KH NO. -570/13" → house
        "road": "road",
        "unit": "unit",
        "level": "level",
        "staircase": "staircase",
        "entrance": "entrance",
        "po_box": "po_box",
        "postcode": "postcode",
        "suburb": "suburb",
        "city_district": "city_district",
        "city": "city",
        "state_district": "state_district",
        "state": "state",
        "country_region": "country_region",
        "country": "country",
        "near": "near",
    }
    for component, label in pairs:
        key = LABEL_MAP.get(label)
        if key and key not in out:
            out[key] = component.lower() if component else None
    return out


def parse_address_components_joined(addr: str) -> dict:
    """Like parse_address_components but allows multiple values per key,
    joined with ' | ' (for components like road with multiple parts)."""
    if not addr:
        return {}
    if not _libpostal_available():
        return {}
    try:
        from postal.parser import parse_address
        pairs = parse_address(addr)
    except Exception:  # noqa: BLE001
        return {}
    LABEL_MAP = {
        "house_number": "house_number",
        "house": "house_number",
        "road": "road",
        "unit": "unit",
        "level": "level",
        "staircase": "staircase",
        "entrance": "entrance",
        "po_box": "po_box",
        "postcode": "postcode",
        "suburb": "suburb",
        "city_district": "city_district",
        "city": "city",
        "state_district": "state_district",
        "state": "state",
        "country_region": "country_region",
        "country": "country",
        "near": "near",
    }
    out: dict = {}
    for component, label in pairs:
        key = LABEL_MAP.get(label)
        if not key:
            continue
        if key in out:
            out[key] = out[key] + " " + component.lower()
        else:
            out[key] = component.lower() if component else ""
    return out


# ---------------------------------------------------------------------------
# Backwards-compatible extract_* helpers (now libpostal-based)
# ---------------------------------------------------------------------------

def extract_postal(addr: str, country: str = "") -> Optional[str]:
    """Extract postal code via libpostal. `country` is accepted for API
    compatibility but libpostal detects the country from the address text."""
    comp = parse_address_components(addr)
    return comp.get("postcode")


def extract_state(addr: str, country: str = "") -> Optional[str]:
    comp = parse_address_components(addr)
    return comp.get("state") or comp.get("state_district")


def extract_city(addr: str, country: str = "") -> Optional[str]:
    comp = parse_address_components(addr)
    return comp.get("city") or comp.get("city_district") or comp.get("suburb")


# ---------------------------------------------------------------------------
# Per-record normalization
# ---------------------------------------------------------------------------

def normalize_record(
    name: str,
    addr: str,
    country: str,
) -> dict:
    """Apply the full normalization pipeline to one record (libpostal-aware)."""
    out: dict = {
        "name_clean": "",
        "name_latin": "",
        "name_tokens": "",
        "name_ngram_key": "",
        "name_dev_ratio": 0.0,
        "addr_clean": "",
        "addr_latin": "",
        "addr_zip": None,
        "addr_state": None,
        "addr_city": None,
        "addr_first_word": "",
        "addr_last_word": "",
        "addr_ngram_key": "",
        "addr_house_number": None,
        "addr_road": None,
        "addr_unit": None,
        "addr_suburb": None,
        "name_missing": 1 if not name else 0,
        "addr_missing": 1 if not addr else 0,
    }

    # ----- name -----
    n_raw = basic_clean(name) if name else ""
    if n_raw:
        n_no_suffix = strip_legal_suffix(n_raw)
        n_no_punct = remove_punct_keep_internal(n_no_suffix)
        out["name_clean"] = MULTI_SPACE_RE.sub(" ", n_no_punct).strip()
        out["name_latin"] = to_latin(out["name_clean"])
        out["name_tokens"] = sorted_tokens(out["name_latin"])
        out["name_ngram_key"] = first_n_chars(out["name_latin"], 4)
        out["name_dev_ratio"] = devanagari_ratio(name or "")
    # ----- address -----
    a_raw = basic_clean(addr) if addr else ""
    if a_raw:
        # NOTE: we intentionally DO NOT call libpostal's expand_address here.
        # It returns 6 multilingual abbreviation expansions per record, which
        # accounts for ~25-40% of libpostal's runtime. Our pipeline does not
        # strictly need it because:
        #   - Transliteration (Devanagari -> Latin) is handled by
        #     src/transliterate.py further down.
        #   - English abbreviations (St, Rd, Pvt, Ltd, Corp, No, etc.) are
        #     handled by our hand-rolled `expand_addr_abbrev` below.
        #   - Downstream blocking/feature engineering uses TF-IDF char
        #     n-grams and rapidfuzz string similarity, which handle
        #     abbreviations implicitly (St and street score high cosine).
        # expand_address may be re-introduced in Phase D on the much-smaller
        # candidate-pair set (10-50M pairs vs 12M normalization rows) if
        # feature engineering shows we need it.
        a_expanded = expand_addr_abbrev(a_raw)
        a_no_punct = remove_punct_keep_internal(a_expanded)
        out["addr_clean"] = MULTI_SPACE_RE.sub(" ", a_no_punct).strip()
        out["addr_latin"] = to_latin(out["addr_clean"])
        out["addr_first_word"] = first_nonstop_word(out["addr_latin"])
        out["addr_last_word"] = last_nonstop_word(out["addr_latin"])
        out["addr_ngram_key"] = first_n_chars(out["addr_latin"], 4)
    # ----- structured fields (libpostal parse) -----
    if addr:
        comp = parse_address_components(addr)
        out["addr_zip"] = comp.get("postcode")
        out["addr_state"] = comp.get("state") or comp.get("state_district")
        out["addr_city"] = comp.get("city") or comp.get("city_district") or comp.get("suburb")
        out["addr_house_number"] = comp.get("house_number")
        out["addr_road"] = comp.get("road")
        out["addr_unit"] = comp.get("unit") or comp.get("po_box")
        out["addr_suburb"] = comp.get("suburb")

    return out


def _normalize_chunk(chunk: list[dict]) -> list[dict]:
    """Normalize a list of (entity_id, business_name, business_address, country) dicts."""
    out: list[dict] = []
    for rec in chunk:
        norm = normalize_record(
            rec.get("business_name", ""),
            rec.get("business_address", ""),
            rec.get("country", ""),
        )
        norm["entity_id"] = rec["entity_id"]
        norm["country"] = rec.get("country", "") or ""
        out.append(norm)
    return out


def normalize_dataframe(df: pl.DataFrame, label: str, n_workers: int = 14) -> pl.DataFrame:
    """Apply normalize_record to every row of a *_source*.tsv DataFrame.

    Parallelized with multiprocessing.Pool — each worker processes a chunk of
    rows. Uses `multiprocessing.get_context("fork")` on Linux for cheap forks.
    """
    import multiprocessing as mp
    import time

    log.info("Normalizing %s (%d rows) with %d workers ...", label, len(df), n_workers)
    t0 = time.time()

    # Materialize rows as plain dicts for pickling (faster than polars IPC).
    rows = df.select(["entity_id", "business_name", "business_address", "country"]).to_dicts()

    chunk_size = max(2000, len(rows) // (n_workers * 8))
    chunks: list[list[dict]] = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    log.info("  split into %d chunks of ~%d rows each", len(chunks), chunk_size)

    out_records: list[dict] = []
    if n_workers <= 1:
        # Sequential path
        for i, ch in enumerate(chunks):
            out_records.extend(_normalize_chunk(ch))
            if (i + 1) % 20 == 0:
                log.info("    %d / %d chunks done (%.1fs)",
                         i + 1, len(chunks), time.time() - t0)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_normalize_chunk, chunks, chunksize=1)):
                out_records.extend(result)
                done = min((i + 1) * chunk_size, len(rows))
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    log.info("    %d / %d rows (%.0f rows/sec, %.1fs elapsed)",
                             done, len(rows), rate, elapsed)

    log.info("  normalization took %.1fs for %d rows (%.0f rows/sec)",
             time.time() - t0, len(rows), len(rows) / max(time.time() - t0, 1e-6))

    schema = {
        "entity_id": pl.Utf8,
        "country": pl.Utf8,
        "name_clean": pl.Utf8,
        "name_latin": pl.Utf8,
        "name_tokens": pl.Utf8,
        "name_ngram_key": pl.Utf8,
        "name_dev_ratio": pl.Float64,
        "addr_clean": pl.Utf8,
        "addr_latin": pl.Utf8,
        "addr_zip": pl.Utf8,
        "addr_state": pl.Utf8,
        "addr_city": pl.Utf8,
        "addr_first_word": pl.Utf8,
        "addr_last_word": pl.Utf8,
        "addr_ngram_key": pl.Utf8,
        "addr_house_number": pl.Utf8,
        "addr_road": pl.Utf8,
        "addr_unit": pl.Utf8,
        "addr_suburb": pl.Utf8,
        "name_missing": pl.Int8,
        "addr_missing": pl.Int8,
    }
    out_df = pl.from_dicts(out_records, schema=schema)
    return out_df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NORMALIZE_TARGETS = [
    ("train_source1", C.TRAIN_SOURCE1, "s1_norm_train.parquet"),
    ("train_source2", C.TRAIN_SOURCE2, "s2_norm_train.parquet"),
    ("train_source3", C.TRAIN_SOURCE3, "s3_norm_train.parquet"),
]

# Test normalization is opt-in. It must only run from `predict.py` /
# `pipeline.py --mode submit` at the final submission step. We never run it
# during development.
SUBMIT_NORMALIZE_TARGETS = [
    ("test_source1",  C.TEST_SOURCE1,  "s1_norm_test.parquet"),
    ("test_source2",  C.TEST_SOURCE2,  "s2_norm_test.parquet"),
    ("test_source3",  C.TEST_SOURCE3,  "s3_norm_test.parquet"),
]


def normalize_submit_files(n_workers: int = 14) -> None:
    """Normalize test inputs ONLY at submission time.

    Explicit, named function — never call this from development scripts.
    """
    if C.TRAIN_ONLY:
        raise RuntimeError(
            "Refusing to run normalize_submit_files while TRAIN_ONLY=True. "
            "Submission generation must explicitly opt-in by setting "
            "config.TRAIN_ONLY = False (or by importing this module under "
            "a separate submission-mode entry point)."
        )
    for label, src_path, out_name in SUBMIT_NORMALIZE_TARGETS:
        df = load_source(src_path)
        norm = normalize_dataframe(df, label, n_workers=n_workers)
        out_path = C.ARTIFACTS_ROOT / out_name
        write_parquet(norm, out_path)


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )

    # Auto-tune worker count based on free RAM. Each libpostal worker loads
    # ~300 MB of language-model data into memory; with 15 GB total and the OS
    # baseline plus polars buffers, 4 workers is a safer cap. Use n_workers=0
    # (default 6) but cap by free memory.
    import os
    avail_gb = os.sysconf('SC_PAGE_SIZE')  # placeholder; we'll read /proc/meminfo
    free_gb = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    free_gb = int(line.split()[1]) / (1024 * 1024)
                    break
    except Exception:  # noqa: BLE001
        free_gb = 8.0

    # Each libpostal worker needs ~600-800 MB peak (libpostal loads ~300 MB
    # of language-model data, plus Python overhead, polars chunk in flight).
    # Reserve 6 GB for OS/parquet I/O/buffers; cap at 4 workers max because
    # we have only 15 GB total and test runs showed 8 workers OOM the box.
    worker_mem_budget = max(0.0, free_gb - 6.0)
    n_workers = max(1, min(4, int(worker_mem_budget / 1.5)))
    log.info("RAM-based worker budget: free=%.1f GB, budget=%.1f GB, n_workers=%d",
             free_gb, worker_mem_budget, n_workers)

    for label, src_path, out_name in NORMALIZE_TARGETS:
        out_path = C.ARTIFACTS_ROOT / out_name
        if out_path.exists():
            log.info("Skipping %s — %s already exists", label, out_path)
            continue
        df = load_source(src_path)
        norm = normalize_dataframe(df, label, n_workers=n_workers)
        write_parquet(norm, out_path)
        # Release memory before next file
        del df, norm
        import gc
        gc.collect()
        log.info("Sample normalized rows for %s:", label)
    return 0


if __name__ == "__main__":
    sys.exit(main())

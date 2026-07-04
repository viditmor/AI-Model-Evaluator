"""
Pure evaluation logic for the Multi-Model LLM Evaluation Platform.

Simplified 4-Metric Evaluation Engine:
1. Answer Accuracy (CrossEncoder)
2. Semantic Similarity (Titan Embeddings V2 Cosine Sim)
3. Response Efficiency (tiktoken length ratio vs Ground Truth)
4. Estimated Cost (informational pricing calculation)
"""

import math
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

from config import RESPONSE_EFFICIENCY_CONFIG, MODEL_PRICING_PER_1K, DEFAULT_PRICING_PER_1K

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning("Failed to initialize tiktoken encoding: %s", e)
    _TIKTOKEN_ENC = None


# ────────────────────────────────────────────────────────────────────
# EMBEDDING PREPROCESSING — used only by Semantic Similarity
# ────────────────────────────────────────────────────────────────────

import json
import re

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")

# Markdown formatting patterns — order matters (most specific first)
_MD_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Images: ![alt](url) → alt
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # Links: [text](url) → text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # Bold/italic combos: ***text*** or ___text___
    (re.compile(r"(\*{3}|_{3})(.+?)\1"), r"\2"),
    # Bold: **text** or __text__
    (re.compile(r"(\*{2}|_{2})(.+?)\1"), r"\2"),
    # Italic: *text* or _text_  (word-boundary guarded to avoid false positives)
    (re.compile(r"(?<!\w)(\*|_)(?!\s)(.+?)(?<!\s)\1(?!\w)"), r"\2"),
    # Strikethrough: ~~text~~
    (re.compile(r"~~(.+?)~~"), r"\1"),
    # Headings: # … ###### at start of line
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # Blockquotes: > at start of line (possibly nested)
    (re.compile(r"^(?:>\s*)+", re.MULTILINE), ""),
    # Horizontal rules: ---, ***, ___ (alone on a line)
    (re.compile(r"^[\s]*([-*_])\s*\1\s*\1[\s\-\*_]*$", re.MULTILINE), ""),
    # Unordered list markers: -, *, + at start of line
    (re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE), ""),
    # Numbered list prefixes: 1. or 1) at start of line
    (re.compile(r"^[\s]*\d+[\.\)]\s+", re.MULTILINE), ""),
    # Inline code: `code`
    (re.compile(r"`([^`]+)`"), r"\1"),
]

# Whitespace normalisation
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MULTI_SPACES_RE = re.compile(r"[^\S\n]{2,}")  # 2+ non-newline whitespace chars


def _json_to_readable(obj: object, prefix: str = "") -> str:
    """Recursively convert a parsed JSON object into 'Key: Value' lines.

    Nested dicts produce indented sub-keys; lists are numbered.
    """
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            label = key.replace("_", " ").replace("-", " ").title()
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{label}:")
                lines.append(_json_to_readable(value, prefix + "  "))
            else:
                lines.append(f"{prefix}{label}: {value}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj, 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{idx}.")
                lines.append(_json_to_readable(item, prefix + "  "))
            else:
                lines.append(f"{prefix}{idx}. {item}")
    else:
        lines.append(f"{prefix}{obj}")
    return "\n".join(lines)


def normalize_for_evaluation(text: str) -> str:
    """Preprocess text before evaluation metrics are calculated.

    Performs:
      1. Strip Markdown code fences (```json, ```, etc.).
      2. Remove Markdown formatting (bold, italic, headers, links, images,
         blockquotes, list markers, numbered list prefixes, inline code).
      3. Normalise whitespace.
      4. If valid JSON, convert into a normalized human-readable representation.
      5. Otherwise return the cleaned plain text.
    """
    if not text or not isinstance(text, str):
        return str(text or "")
    if text.startswith("__ERROR__"):
        return text

    # Step 1 — remove code fences
    cleaned = _CODE_FENCE_RE.sub("", text)

    # Step 2 — strip Markdown formatting
    for pattern, replacement in _MD_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # Step 3 — normalise whitespace
    cleaned = _MULTI_SPACES_RE.sub(" ", cleaned)
    cleaned = _MULTI_BLANK_LINES_RE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()

    # Step 4 — attempt JSON normalisation
    try:
        parsed = json.loads(cleaned)
        readable = _json_to_readable(parsed)
        if readable.strip():
            return readable.strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Step 5 — return cleaned plain text
    return cleaned


def prepare_embedding_text(text: str) -> str:
    """Backward-compatible alias for normalize_for_evaluation."""
    return normalize_for_evaluation(text)


# ────────────────────────────────────────────────────────────────────
# METRIC 1 — Answer Accuracy (CrossEncoder)
# ────────────────────────────────────────────────────────────────────

def score_cross_encoder(output: str,
                        ground_truth: str,
                        ce_model: CrossEncoder) -> float:
    """
    Compare generated response against Ground Truth using CrossEncoder.
    Primary evaluation metric. Evaluates semantic content on normalized text.
    """
    if not output.strip() or output.startswith("__ERROR__"):
        return 0.0
    norm_out = normalize_for_evaluation(output)
    norm_gt = normalize_for_evaluation(ground_truth)
    if not norm_out.strip():
        return 0.0
    raw = ce_model.predict([(norm_out, norm_gt)])
    normed = (float(raw[0]) + 1) / 2  # [-1, 1] → [0, 1]
    return round(max(0.0, min(1.0, normed)), 4)


# ────────────────────────────────────────────────────────────────────
# METRIC 2 — Semantic Similarity (Titan Cosine Sim)
# ────────────────────────────────────────────────────────────────────

def score_embedding_similarity(output_embedding: list,
                               gt_embedding: list) -> float:
    """
    Compare Titan embedding vectors of Ground Truth and Generated Response
    using Cosine Similarity.
    """
    if not output_embedding or not gt_embedding:
        return 0.0
    sim = cosine_similarity([output_embedding], [gt_embedding])[0][0]
    return round(max(0.0, min(1.0, float(sim))), 4)


# ────────────────────────────────────────────────────────────────────
# METRIC 3 — Response Efficiency (Length vs Ground Truth)
# ────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding).

    Used to tokenize the Ground Truth once per evaluation so that
    Response Efficiency has a consistent ideal-length baseline.
    Falls back to whitespace splitting if tiktoken is unavailable.
    """
    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return len(text.split())


def score_length_appropriateness(output_tokens: int, gt_tokens: int) -> float:
    """Evaluate whether the response length is appropriate compared to Ground Truth.

    Tolerance zone: If 0.70 <= Ratio <= 1.30, score = 1.0 (100%).
    Outside this region, the score decreases smoothly and gradually without sharp drops.
    """
    if output_tokens < 1:
        return 0.0
    gt_tokens = max(1, gt_tokens)
    ratio = float(output_tokens) / float(gt_tokens)

    if 0.70 <= ratio <= 1.30:
        return 1.0
    elif ratio > 1.30:
        # Smooth exponential decay above 1.30
        return max(0.0, min(1.0, math.exp(-0.55 * (ratio - 1.30))))
    else:
        # Smooth exponential decay below 0.70
        return max(0.0, min(1.0, math.exp(-3.8 * (0.70 - ratio))))


def score_information_density(output_text: str) -> float:
    """Evaluate whether the response has high information density without unnecessary repetition.

    Uses a lightweight repetition detector checking word 5-gram uniqueness and clause/line uniqueness.
    Does not penalize legitimate structure or normal phrasing.
    """
    if not output_text or not output_text.strip():
        return 0.0

    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", output_text.lower())
    if len(words) < 15:
        return 1.0

    # Word 5-gram uniqueness ratio
    ngrams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
    ngram_ratio = len(set(ngrams)) / float(len(ngrams)) if ngrams else 1.0

    # Clause / line uniqueness ratio
    raw_clauses = re.split(r"[\n.!?]+", output_text)
    clauses = [c.strip().lower() for c in raw_clauses if len(c.strip().split()) >= 4]
    clause_ratio = len(set(clauses)) / float(len(clauses)) if len(clauses) >= 3 else 1.0

    # Combine ratios
    raw_density = 0.65 * ngram_ratio + 0.35 * clause_ratio
    return max(0.0, min(1.0, raw_density * 1.06))


def score_completion(output_text: str, output_tokens: int, gt_tokens: int = 0) -> float:
    """Ensure the response is complete and not truncated or abruptly ended.

    Penalizes incomplete outputs, trailing conjunctions/commas without punctuation,
    malformed/truncated JSON blocks, or responses with fewer than 5 tokens.
    """
    if output_tokens < 5 or not output_text or not output_text.strip():
        return 0.0
    if output_text.startswith("__ERROR__"):
        return 0.0

    text = output_text.strip()

    # Relative length completeness check for severe fragments
    if gt_tokens >= 20 and output_tokens < 15 and float(output_tokens) < 0.20 * float(gt_tokens):
        return 0.0

    score = 1.0

    # Check terminal punctuation / structure
    terminal_chars = set(".!?\"'}]>*`~\n")
    if text[-1] not in terminal_chars:
        trailing_words = {"and", "or", "but", "with", "for", "to", "of", "in", "the", "a", "an", "is", "are", "by", "as", "at"}
        last_word = re.findall(r"\b[a-zA-Z]+\b", text.split()[-1].lower())
        last_word_str = last_word[0] if last_word else ""
        if text[-1] in ",:;-" or last_word_str in trailing_words:
            score *= 0.30
        else:
            score *= 0.65

    # Check for unclosed JSON block
    cleaned_start = _CODE_FENCE_RE.sub("", text).strip()
    if cleaned_start.startswith("{") or cleaned_start.startswith("["):
        try:
            json.loads(cleaned_start)
        except Exception:
            score *= 0.30

    return max(0.0, min(1.0, score))


def score_response_efficiency(output_tokens: int,
                              gt_tokens: int,
                              output_text: str = "",
                              sigma: float = 0.5) -> float:
    """Measure Response Efficiency using three modular sub-metrics:
    1. Length Appropriateness (40%)
    2. Information Density (40%)
    3. Completion Check (20%)

    Returns composite value clamped to [0.0, 1.0].
    """
    if output_tokens < 5 or (output_text and output_text.startswith("__ERROR__")):
        return 0.0

    length_score = score_length_appropriateness(output_tokens, gt_tokens)
    density_score = score_information_density(output_text) if output_text else 1.0
    completion_score = score_completion(output_text, output_tokens, gt_tokens) if output_text else 1.0

    final_score = (
        0.40 * length_score +
        0.40 * density_score +
        0.20 * completion_score
    )
    return round(max(0.0, min(1.0, final_score)), 4)


# ────────────────────────────────────────────────────────────────────
# METRIC 4 — Estimated Cost calculation (Informational)
# ────────────────────────────────────────────────────────────────────

def calculate_cost(model_id: str,
                   input_tokens: int,
                   output_tokens: int) -> dict[str, float]:
    """
    Calculate estimated inference cost based on model pricing per 1,000 tokens.
    Never contributes to composite score.
    """
    mid_lower = model_id.lower()
    in_price_1k, out_price_1k = DEFAULT_PRICING_PER_1K

    for key, (inp, outp) in MODEL_PRICING_PER_1K.items():
        if key in mid_lower:
            in_price_1k, out_price_1k = inp, outp
            break

    in_cost = (input_tokens / 1000.0) * in_price_1k
    out_cost = (output_tokens / 1000.0) * out_price_1k
    tot_cost = in_cost + out_cost

    return {
        "input_cost": round(in_cost, 6),
        "output_cost": round(out_cost, 6),
        "total_cost": round(tot_cost, 6),
    }


# ────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────

def evaluate_single(prompt: str,
                    ground_truth: str,
                    model_outputs: dict[str, str],
                    model_token_usage: dict[str, dict],
                    gt_embedding: list,
                    output_embeddings: dict[str, list],
                    ce_model: CrossEncoder,
                    sigma: float = 0.5,
                    weights: dict[str, float] | None = None) -> dict:
    """
    Run evaluation pipeline across all model outputs for the 4 core metrics.
    """
    default_w = {
        "cross_encoder": 0.45,
        "embedding_similarity": 0.30,
        "response_efficiency": 0.15,
    }
    w = weights if weights else default_w

    # Normalize weights so they sum to 1.0 (100%)
    total_w = sum(w.get(k, 0.0) for k in ["cross_encoder", "embedding_similarity", "response_efficiency"])
    if total_w <= 0:
        w_ce, w_sim, w_eff = 0.50, 0.3333, 0.1667
    else:
        w_ce  = w.get("cross_encoder", 0.0) / total_w
        w_sim = w.get("embedding_similarity", 0.0) / total_w
        w_eff = w.get("response_efficiency", 0.0) / total_w

    # Compute Ground Truth token count once — used by every model's
    # Response Efficiency score.  Model output tokens come from Bedrock
    # directly (matching the values shown in the frontend).
    gt_tokens = count_tokens(ground_truth)

    results = []
    best_composite = -1.0
    winner_id = None

    for model_id, output_text in model_outputs.items():
        is_error = output_text.startswith("__ERROR__")
        token_data = model_token_usage.get(model_id, {})
        in_tokens = token_data.get("input_tokens", 0)
        out_tokens = token_data.get("output_tokens", 0)

        if is_error:
            ce_score = 0.0
            sim_score = 0.0
            eff_score = 0.0
        else:
            norm_output = normalize_for_evaluation(output_text)
            ce_score = score_cross_encoder(norm_output, ground_truth, ce_model)
            sim_score = score_embedding_similarity(output_embeddings.get(model_id, []), gt_embedding)
            eff_score = score_response_efficiency(out_tokens, gt_tokens, norm_output, sigma)

        composite = round(ce_score * w_ce + sim_score * w_sim + eff_score * w_eff, 4)
        cost_info = calculate_cost(model_id, in_tokens, out_tokens)

        results.append({
            "model_id": model_id,
            "output": output_text,
            "cross_encoder": ce_score,
            "embedding_similarity": sim_score,
            "response_efficiency": eff_score,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "input_cost": cost_info["input_cost"],
            "output_cost": cost_info["output_cost"],
            "total_cost": cost_info["total_cost"],
            "composite": composite,
        })

        if composite > best_composite:
            best_composite = composite
            winner_id = model_id

    return {
        "results": results,
        "winner": winner_id,
    }

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

TokenCounts = Tuple[int, int, int]

# Pricing tables for Gemini batch API (USD per 1M tokens).
GEMINI_BATCH_RATES = {
    "gemini-2.5-pro": {
        "threshold": 200_000,  # prompts above this use the higher tier
        "input_low": 0.625,
        "input_high": 1.25,
        "output_low": 5.0,
        "output_high": 7.5,
    },
    "gemini-2.5-flash": {
        "input_default": 0.15,          # text / image / video
        "input_audio": 0.50,
        "cached_default": 0.03,         # discounted cached rate per docs
        "cached_audio": 0.03,
        "output": 1.25,
    },
}


def usage_to_dict(token_usage: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of usage metadata to a serializable dictionary."""
    if token_usage is None:
        return None
    if isinstance(token_usage, dict):
        return token_usage
    if hasattr(token_usage, "to_dict"):
        return token_usage.to_dict()

    data: Dict[str, Any] = {}
    candidate_keys = [
        "input_token_count",
        "output_token_count",
        "total_token_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_content_token_count",
        "thinking_token_count",
    ]
    for key in candidate_keys:
        if hasattr(token_usage, key):
            value = getattr(token_usage, key)
            if value is not None:
                data[key] = value
    return data or None


def extract_token_counts(usage: Optional[Dict[str, Any]]) -> TokenCounts:
    """Returns (input_tokens, output_tokens, total_tokens) from usage metadata."""
    if not usage:
        return 0, 0, 0

    def _get(keys: Tuple[str, ...]) -> int:
        for key in keys:
            if key in usage and usage[key] is not None:
                try:
                    return int(usage[key])
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = _get(
        (
            "input_token_count",
            "input_tokens",
            "prompt_token_count",
            "inputTokenCount",
            "promptTokenCount",
        )
    )
    output_tokens = _get(
        (
            "output_token_count",
            "output_tokens",
            "candidates_token_count",
            "outputTokenCount",
            "candidatesTokenCount",
        )
    )
    total_tokens = _get(
        (
            "total_token_count",
            "total_tokens",
            "totalTokenCount",
        )
    )
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def extract_cached_tokens(usage: Optional[Dict[str, Any]]) -> int:
    """Extracts the cached content token count if present."""
    if not usage:
        return 0
    for key in ("cached_content_token_count", "cachedContentTokenCount"):
        if key in usage and usage[key] is not None:
            try:
                return int(usage[key])
            except (TypeError, ValueError):
                continue
    return 0


def _select_rates(model_id: str, modality: Optional[str], input_tokens: int) -> Optional[Dict[str, float]]:
    """Returns the applicable input/output price-per-million for a given model."""
    model_key = model_id.lower()
    if model_key.startswith("models/"):
        model_key = model_key[len("models/") :]
    if ":" in model_key:
        model_key = model_key.split(":", 1)[0]
    if model_key.startswith("gemini-2.5-pro"):
        rates = GEMINI_BATCH_RATES["gemini-2.5-pro"]
        threshold = rates["threshold"]
        high_tier = input_tokens > threshold
        return {
            "input_rate": rates["input_high"] if high_tier else rates["input_low"],
            "output_rate": rates["output_high"] if high_tier else rates["output_low"],
            "cached_input_rate": rates["input_high"] if high_tier else rates["input_low"],
            "tier": "tier2" if high_tier else "tier1",
        }

    if model_key.startswith("gemini-2.5-flash"):
        rates = GEMINI_BATCH_RATES["gemini-2.5-flash"]
        is_audio = (modality or "").lower() == "audio"
        input_rate = rates["input_audio"] if is_audio else rates["input_default"]
        cached_rate = rates["cached_audio"] if is_audio else rates["cached_default"]
        return {
            "input_rate": input_rate,
            "cached_input_rate": cached_rate,
            "output_rate": rates["output"],
            "tier": "audio" if is_audio else "default",
        }

    return None


def estimate_cost_usd(
    model_id: str,
    modality: Optional[str],
    usage: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Computes the API cost for a single sample or cache event, if pricing data is available."""
    usage_dict = usage_to_dict(usage)
    input_tokens, output_tokens, total_tokens = extract_token_counts(usage_dict)
    if input_tokens == 0 and output_tokens == 0:
        return None

    rates = _select_rates(model_id, modality, input_tokens)
    if rates is None:
        return None

    cached_tokens = min(extract_cached_tokens(usage_dict), input_tokens)
    cached_rate = rates.get("cached_input_rate", rates["input_rate"])
    uncached_tokens = max(input_tokens - cached_tokens, 0)

    input_cost = (uncached_tokens / 1_000_000) * rates["input_rate"]
    cached_cost = (cached_tokens / 1_000_000) * cached_rate
    output_cost = (output_tokens / 1_000_000) * rates["output_rate"]

    total_cost = input_cost + cached_cost + output_cost

    return {
        "usd": total_cost,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_rate_per_million": rates["input_rate"],
        "cached_input_rate_per_million": cached_rate,
        "output_rate_per_million": rates["output_rate"],
        "pricing_tier": rates.get("tier", "default"),
    }

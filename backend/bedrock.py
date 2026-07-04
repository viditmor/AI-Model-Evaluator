"""
AWS Bedrock interactions for the Multi-Model LLM Evaluation Platform.

All functions receive boto3 clients as parameters — no global state.
Uses the Converse API as a universal model invoker.
"""

import json
import logging
import random
import re
import time

from config import EMBED_MODEL, RETRY_ATTEMPTS, RETRY_BASE, RETRY_MAX_WAIT

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# PROVIDER-SPECIFIC INFERENCE CONFIGURATION
# ────────────────────────────────────────────────────────────────────
# Each provider only receives inference parameters it actually supports.
# This avoids ValidationException from providers that reject unknown
# parameters (e.g. Mistral rejects "temperature").

PROVIDER_INFERENCE_SUPPORT: dict[str, set[str]] = {
    "anthropic": {"maxTokens", "temperature", "topP", "stopSequences"},
    "amazon":    {"maxTokens", "temperature", "topP", "stopSequences"},
    "meta":      {"maxTokens", "temperature", "topP"},
    "cohere":    {"maxTokens", "temperature", "topP"},
    "ai21":      {"maxTokens", "temperature", "topP", "stopSequences"},
    "mistral":   {"maxTokens"},
}

# Safe fallback for providers not listed above — maxTokens only.
_DEFAULT_SUPPORTED_PARAMS: set[str] = {"maxTokens"}


def _get_provider(model_id: str) -> str:
    """Extract the provider prefix from a Bedrock model ID.

    Handles both plain IDs ('anthropic.claude-…') and inference-profile
    IDs ('us.anthropic.claude-…') by scanning dot-separated segments
    for a known provider name.

    Returns the lowercase provider name, or 'unknown' if unrecognised.
    """
    segments = model_id.lower().split(".")
    for seg in segments:
        if seg in PROVIDER_INFERENCE_SUPPORT:
            return seg
    # Fallback: first segment (best guess)
    return segments[0] if segments else "unknown"


def _build_inference_config(
    model_id: str,
    *,
    max_tokens: int = 1000,
    temperature: float | None = 0.1,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
) -> dict:
    """Build a provider-safe inferenceConfig dict.

    Only includes parameters that the provider is known to support.
    Unknown providers receive maxTokens only (safest common denominator).
    """
    provider = _get_provider(model_id)
    supported = PROVIDER_INFERENCE_SUPPORT.get(provider, _DEFAULT_SUPPORTED_PARAMS)

    config: dict = {}
    if "maxTokens" in supported:
        config["maxTokens"] = max_tokens
    if "temperature" in supported and temperature is not None:
        config["temperature"] = temperature
    if "topP" in supported and top_p is not None:
        config["topP"] = top_p
    if "stopSequences" in supported and stop_sequences is not None:
        config["stopSequences"] = stop_sequences

    return config


# ────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────

# Pattern: valid Converse model IDs end with  vN:M  (e.g. v1:0).
# list-foundation-models may append extra segments like :48k, :200k.
# We strip everything after the version:digit portion.
_MODEL_ID_RE = re.compile(
    r"^(?P<base>.+?-v\d+:\d+)(?::.+)?$"
)


def _clean_model_id(raw_id: str) -> str:
    """Strip context-length variant suffixes from a Bedrock model ID.

    Examples:
        'anthropic.claude-3-haiku-20240307-v1:0:48k'  → 'anthropic.claude-3-haiku-20240307-v1:0'
        'anthropic.claude-3-sonnet-20240229-v1:0'      → (unchanged)
        'amazon.nova-pro-v1:0'                         → (unchanged)
    """
    m = _MODEL_ID_RE.match(raw_id)
    return m.group("base") if m else raw_id


def _build_inference_profile_map(bedrock_client) -> dict[str, str]:
    """Build a mapping from base model ID → system-defined inference profile ID.

    Calls list_inference_profiles(typeEquals='SYSTEM_DEFINED') and maps each
    profile's underlying model(s) to the profile's inferenceProfileId.

    We prefer US-region profiles (us.*) when available.

    Returns:
        dict mapping e.g.
          'anthropic.claude-opus-4-5-20251101-v1:0' → 'us.anthropic.claude-opus-4-5-20251101-v1:0'
    """
    profile_map: dict[str, str] = {}
    try:
        paginator = bedrock_client.get_paginator("list_inference_profiles")
        for page in paginator.paginate(typeEquals="SYSTEM_DEFINED"):
            for profile in page.get("inferenceProfileSummaries", []):
                profile_id = profile["inferenceProfileId"]
                # Each profile lists the foundation models it wraps
                for model_ref in profile.get("models", []):
                    base_model_id = model_ref.get("modelArn", "").split("/")[-1]
                    if not base_model_id:
                        continue
                    # Prefer 'us.' prefixed profiles over others
                    existing = profile_map.get(base_model_id)
                    if existing is None or profile_id.startswith("us."):
                        profile_map[base_model_id] = profile_id
    except Exception as e:
        logger.warning("Failed to list inference profiles: %s. "
                       "Models requiring profiles will be excluded.", e)
    return profile_map


# ────────────────────────────────────────────────────────────────────
# LIST AVAILABLE TEXT-GENERATION MODELS
# ────────────────────────────────────────────────────────────────────

def list_text_generation_models(bedrock_client) -> list[dict]:
    """
    List all text-generation foundation models available in Bedrock.

    Filters to models that have 'TEXT' in both inputModalities and
    outputModalities. Excludes models whose outputModalities contain
    'IMAGE' or 'EMBEDDING' (i.e., image generators, embedding models).

    Handles two known issues:
      1. Strips context-length suffixes (e.g. ':48k') that the Converse
         API does not accept.
      2. For models that only support INFERENCE_PROFILE invocation, resolves
         the system-defined inference profile ID to use instead.

    Args:
        bedrock_client: boto3 Bedrock client (not runtime).

    Returns:
        Sorted list of dicts with keys:
          modelId, modelName, providerName, invokeModelId.
    """
    response = bedrock_client.list_foundation_models()
    model_summaries = response.get("modelSummaries", [])

    # Pre-build the inference-profile lookup
    profile_map = _build_inference_profile_map(bedrock_client)

    text_models: list[dict] = []
    seen_ids: set[str] = set()

    for model in model_summaries:
        input_modalities = model.get("inputModalities", [])
        output_modalities = model.get("outputModalities", [])

        # Must support TEXT input and TEXT output
        if "TEXT" not in input_modalities or "TEXT" not in output_modalities:
            continue

        # Exclude image generators and embedding models
        if "IMAGE" in output_modalities or "EMBEDDING" in output_modalities:
            continue

        # Exclude LEGACY models — they return "Access denied" unless
        # the user has been actively using them in the last 30 days.
        lifecycle = model.get("modelLifecycle", {})
        if lifecycle.get("status") == "LEGACY":
            continue

        raw_id = model["modelId"]
        clean_id = _clean_model_id(raw_id)

        # Deduplicate (e.g. model-v1:0 and model-v1:0:48k become the same)
        if clean_id in seen_ids:
            continue
        seen_ids.add(clean_id)

        # Determine the ID to actually pass to Converse
        inference_types = model.get("inferenceTypesSupported", [])

        if "ON_DEMAND" in inference_types:
            # Model can be invoked directly with the cleaned ID
            invoke_id = clean_id
        elif clean_id in profile_map:
            # Model requires an inference profile — use the mapped one
            invoke_id = profile_map[clean_id]
        else:
            # Model can't be invoked (no ON_DEMAND, no profile found) — skip
            logger.info(
                "Skipping %s — no ON_DEMAND support and no inference profile found.",
                clean_id,
            )
            continue

        text_models.append({
            "modelId": clean_id,
            "modelName": model["modelName"],
            "providerName": model["providerName"],
            "invokeModelId": invoke_id,
        })

    # Sort by provider then model name
    text_models.sort(key=lambda m: (m["providerName"], m["modelName"]))
    return text_models


# ────────────────────────────────────────────────────────────────────
# INVOKE MODEL — Converse API (universal format)
# ────────────────────────────────────────────────────────────────────

def invoke_model(
    runtime_client,
    model_id: str,
    user_prompt: str,
    system_prompt: str | None = None,
    output_parser: str | None = None,
    output_schema: str | None = None,
) -> dict:
    """
    Invoke a Bedrock model using the Converse API.

    The Converse API provides a single unified format that works across
    Claude, Llama, Mistral, Nova, Titan, Cohere, and all other Bedrock
    models.

    Messages are built dynamically:
      - If *system_prompt* is provided it is sent as a system message.
      - *user_prompt* is always sent as the user message.
      - If *output_parser* is provided, format instructions are appended
        to the user message text.

    Includes exponential backoff + full jitter retry logic.

    Args:
        runtime_client: boto3 bedrock-runtime client.
        model_id: The Bedrock model ID to invoke.
        user_prompt: The user prompt to send.
        system_prompt: Optional system-level instructions.
        output_parser: Optional output format instructions.
        output_schema: Optional JSON schema for structured output.

    Returns:
        Dict with keys 'text' (str), 'input_tokens' (int), 'output_tokens' (int).

    Raises:
        The last exception encountered if all retry attempts fail.
    """
    # ── Build user message text ──────────────────────────────────────
    user_text = user_prompt
    schema_or_parser = output_schema or output_parser
    if schema_or_parser:
        user_text += (
            "\n\n[Output Schema Instruction]\n"
            "You MUST output valid JSON strictly conforming to this JSON schema:\n"
            f"{schema_or_parser}"
        )

    # ── Build Converse kwargs ────────────────────────────────────────
    converse_kwargs: dict = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": _build_inference_config(model_id),
    }
    if system_prompt:
        converse_kwargs["system"] = [{"text": system_prompt}]

    # ── Call with retry ──────────────────────────────────────────────
    last_err: Exception | None = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = runtime_client.converse(**converse_kwargs)
            result_text = response["output"]["message"]["content"][0]["text"]
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            return {
                "text": result_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            last_err = e
            cap = min(RETRY_MAX_WAIT, RETRY_BASE * (2 ** attempt))
            wait = random.uniform(0, cap)
            logger.warning(
                "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                model_id, attempt + 1, RETRY_ATTEMPTS, e, wait,
            )
            time.sleep(wait)

    raise last_err  # type: ignore[misc]


# ────────────────────────────────────────────────────────────────────
# TITAN EMBEDDING
# ────────────────────────────────────────────────────────────────────

def get_titan_embedding(runtime_client, text: str) -> list:
    """
    Fetch a Titan text embedding vector for the given text.

    Args:
        runtime_client: boto3 bedrock-runtime client.
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    resp = runtime_client.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]

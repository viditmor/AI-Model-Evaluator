"""
Configuration constants for the Multi-Model LLM Evaluation Platform.

Simplified 4-Metric Evaluation Pipeline:
  Final = 0.45 × Answer Accuracy      (CrossEncoder cross-encoder/stsb-roberta-large)
        + 0.30 × Semantic Similarity  (Amazon Titan Embeddings V2 Cosine Sim)
        + 0.15 × Response Efficiency  (Gaussian ratio against Ground Truth length via tiktoken)
Note: Weights are normalized so that 45 + 30 + 15 = 90% scales to 100%.
Estimated Cost is informational only and never contributes to composite score.
"""

# AWS Region
REGION: str = "us-east-1"

# Titan Embedding Model
EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"

# Composite score relative weights
WEIGHTS: dict[str, float] = {
    "cross_encoder":        0.45,
    "embedding_similarity": 0.30,
    "response_efficiency":  0.15,
}

# Response Efficiency configuration
RESPONSE_EFFICIENCY_CONFIG: dict = {
    "ideal_ratio": 1.0,
    "default_sigma": 0.5,
}

# Thread pool size for parallel model invocations
MAX_WORKERS: int = 20

# Retry configuration — exponential backoff + full jitter
RETRY_ATTEMPTS: int = 4
RETRY_BASE: float = 1.5
RETRY_MAX_WAIT: float = 30

# Inference pricing per 1,000 tokens (in USD) for AWS Bedrock models.
# Values are structured as (input_price_per_1k, output_price_per_1k).
MODEL_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    # Anthropic Claude
    "claude-3-5-sonnet": (0.00300, 0.01500),
    "claude-3-sonnet":   (0.00300, 0.01500),
    "claude-3-haiku":    (0.00025, 0.00125),
    "claude-3-5-haiku":  (0.00080, 0.00400),
    "claude-3-opus":     (0.01500, 0.07500),
    # Amazon Nova
    "nova-pro":          (0.00080, 0.00320),
    "nova-lite":         (0.00006, 0.00024),
    "nova-micro":        (0.000035, 0.00014),
    # Amazon Titan Text
    "titan-text-express": (0.00020, 0.00060),
    "titan-text-lite":    (0.00015, 0.00020),
    # Meta Llama
    "llama3-70b":        (0.00072, 0.00072),
    "llama3-1-70b":      (0.00072, 0.00072),
    "llama3-2-90b":      (0.00072, 0.00072),
    "llama3-8b":         (0.00015, 0.00015),
    "llama3-1-8b":       (0.00015, 0.00015),
    "llama3-2-3b":       (0.00015, 0.00015),
    "llama3-2-1b":       (0.00010, 0.00010),
    "llama3-405b":       (0.00240, 0.00240),
    # Mistral
    "mistral-large":     (0.00200, 0.00600),
    "mistral-small":     (0.00020, 0.00060),
    # Cohere Command R
    "command-r-plus":    (0.00300, 0.01500),
    "command-r":         (0.00050, 0.00150),
    # AI21
    "jamba-1-5-large":   (0.00200, 0.00800),
    "jamba-1-5-mini":    (0.00020, 0.00040),
}

# Default fallback pricing if exact model substring is not matched
DEFAULT_PRICING_PER_1K: tuple[float, float] = (0.00100, 0.00300)

"""
Pydantic schemas for the Multi-Model LLM Evaluation Platform API.
"""

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    """Request body for the /evaluate endpoint."""

    user_prompt: str = Field(..., min_length=1, description="The actual task or question for the LLM.")
    system_prompt: str | None = Field(None, description="Optional system prompt defining model behaviour.")
    output_parser: str | None = Field(None, description="Optional output format instructions.")
    output_schema: str | None = Field(None, description="Optional JSON schema for structured output validation.")
    ground_truth: str = Field(..., min_length=1, description="The reference answer to evaluate against.")
    model_ids: list[str] = Field(..., min_length=1, description="List of Bedrock model IDs to evaluate.")
    weights: dict[str, float] | None = Field(
        None,
        description="Optional custom weights for scoring metrics. Keys: cross_encoder, embedding_similarity, response_efficiency."
    )
    sigma: float | None = Field(None, description="Optional custom Gaussian tolerance for Response Efficiency scoring.")


class ModelResult(BaseModel):
    """Evaluation result for a single model focusing on the 4 core metrics."""

    model_id: str
    model_name: str
    provider: str
    output: str
    cross_encoder: float = Field(..., description="Answer Accuracy score (0.0 to 1.0)")
    embedding_similarity: float = Field(..., description="Semantic Similarity score (0.0 to 1.0)")
    response_efficiency: float = Field(..., description="Response Efficiency score (0.0 to 1.0)")
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    composite: float = Field(..., description="Overall normalized composite score (0.0 to 1.0)")


class EvaluateResponse(BaseModel):
    """Response body for the /evaluate endpoint."""

    user_prompt: str
    ground_truth: str
    winner: ModelResult
    results: list[ModelResult]


class BedrockModel(BaseModel):
    """A single Bedrock foundation model descriptor."""

    model_id: str
    model_name: str
    provider: str
    invoke_model_id: str = Field(
        ...,
        description="The ID to actually pass to the Converse API. "
        "May be an inference-profile ID or a cleaned model ID.",
    )


class ModelsResponse(BaseModel):
    """Response body for the /models endpoint."""

    models: list[BedrockModel]
    providers: list[str]

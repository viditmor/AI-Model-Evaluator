"""
FastAPI application for the Multi-Model LLM Evaluation Platform.

Startup loads boto3 clients, CrossEncoder, and ThreadPoolExecutor into app.state.
Simplified 4-metric evaluation engine.
"""

import json
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder

from config import REGION, MAX_WORKERS, RESPONSE_EFFICIENCY_CONFIG
from models import (
    BedrockModel,
    EvaluateRequest,
    EvaluateResponse,
    ModelResult,
    ModelsResponse,
)
import bedrock
import evaluation
from evaluation import normalize_for_evaluation, prepare_embedding_text

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# LIFESPAN — startup / shutdown
# ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler."""
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Loading boto3 clients (region=%s)...", REGION)
    app.state.bedrock_client = boto3.client("bedrock", region_name=REGION)
    app.state.runtime_client = boto3.client("bedrock-runtime", region_name=REGION)

    # ── Pre-build model list & invoke ID map ─────────────────────────
    logger.info("Building model list and invoke ID map...")
    try:
        raw_models = bedrock.list_text_generation_models(app.state.bedrock_client)
    except Exception as e:
        logger.error("Failed to build model list at startup: %s", e)
        raw_models = []

    models = [
        BedrockModel(
            model_id=m["modelId"],
            model_name=m["modelName"],
            provider=m["providerName"],
            invoke_model_id=m["invokeModelId"],
        )
        for m in raw_models
    ]
    app.state.model_list = models
    app.state.providers = sorted({m.provider for m in models})
    app.state.invoke_id_map = {m.model_id: m.invoke_model_id for m in models}
    logger.info("Model list ready: %d models from %d providers.",
                len(models), len(app.state.providers))

    logger.info("Loading CrossEncoder (stsb-roberta-large)...")
    app.state.ce_model = CrossEncoder("cross-encoder/stsb-roberta-large")

    logger.info("Creating ThreadPoolExecutor (max_workers=%d)...", MAX_WORKERS)
    app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    logger.info("Startup complete ✓")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down ThreadPoolExecutor...")
    app.state.executor.shutdown(wait=False)
    logger.info("Shutdown complete ✓")


# ────────────────────────────────────────────────────────────────────
# APP CREATION
# ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Model LLM Evaluation Platform",
    description="Evaluate and compare LLM outputs using a simplified 4-metric dashboard pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────

def _parse_model_id(model_id: str) -> tuple[str, str]:
    """Extract (provider, model_name) from raw model ID."""
    if "." in model_id:
        provider_raw, name = model_id.split(".", 1)
        provider = provider_raw.capitalize()
    else:
        provider = "Unknown"
        name = model_id
    return provider, name


# ────────────────────────────────────────────────────────────────────
# GET /models
# ────────────────────────────────────────────────────────────────────

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """List all available text-generation Bedrock models."""
    return ModelsResponse(
        models=app.state.model_list,
        providers=app.state.providers,
    )


# ────────────────────────────────────────────────────────────────────
# POST /evaluate
# ────────────────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Evaluate multiple Bedrock models focusing on the 4 core metrics.
    """
    user_prompt = request.user_prompt.strip()
    system_prompt = request.system_prompt.strip() if request.system_prompt else None
    output_parser = request.output_parser.strip() if request.output_parser else None
    ground_truth = request.ground_truth.strip()
    model_ids = request.model_ids

    if not user_prompt:
        raise HTTPException(status_code=422, detail="User prompt must not be empty.")
    if not ground_truth:
        raise HTTPException(status_code=422, detail="Ground truth must not be empty.")
    if not model_ids:
        raise HTTPException(status_code=422, detail="At least one model_id is required.")

    output_schema = request.output_schema.strip() if request.output_schema else None
    schema_to_use = output_schema or output_parser
    if schema_to_use:
        try:
            json.loads(schema_to_use)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Output Schema JSON: {e}")

    runtime_client = app.state.runtime_client
    executor = app.state.executor
    ce_model = app.state.ce_model

    # Resolve display model IDs → invocation IDs
    invoke_id_map: dict[str, str] = app.state.invoke_id_map
    resolved_ids: dict[str, str] = {}
    for mid in model_ids:
        if mid not in invoke_id_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown invoke ID for model: {mid}",
            )
        resolved_ids[mid] = invoke_id_map[mid]

    # Step 1: Fetch GT embedding in parallel (preprocessed for semantic comparison)
    try:
        gt_cleaned = normalize_for_evaluation(ground_truth)
        gt_fut = executor.submit(bedrock.get_titan_embedding, runtime_client, gt_cleaned)
        gt_embedding = gt_fut.result()
    except Exception as e:
        logger.exception("Failed to fetch GT embedding")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding invocation error: {e}",
        )

    # Step 2: Invoke selected models in parallel
    model_outputs: dict[str, str] = {}
    model_token_usage: dict[str, dict] = {}
    future_to_model = {
        executor.submit(
            bedrock.invoke_model, runtime_client, resolved_ids[mid],
            user_prompt, system_prompt, output_parser, schema_to_use,
        ): mid
        for mid in model_ids
    }

    for future in as_completed(future_to_model):
        mid = future_to_model[future]
        try:
            result = future.result()
            model_outputs[mid] = result["text"]
            model_token_usage[mid] = {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }
        except Exception as e:
            logger.warning("Model %s failed: %s", mid, e)
            model_outputs[mid] = f"__ERROR__: {e}"
            model_token_usage[mid] = {"input_tokens": 0, "output_tokens": 0}

    # Step 3: Fetch output embeddings in parallel (preprocessed for semantic comparison)
    output_embeddings: dict[str, list] = {}
    emb_future_to_model = {}
    for mid, output_text in model_outputs.items():
        if not output_text.startswith("__ERROR__"):
            cleaned_output = normalize_for_evaluation(output_text)
            emb_future_to_model[
                executor.submit(
                    bedrock.get_titan_embedding, runtime_client, cleaned_output
                )
            ] = mid

    for future in as_completed(emb_future_to_model):
        mid = emb_future_to_model[future]
        try:
            output_embeddings[mid] = future.result()
        except Exception as e:
            logger.warning("Embedding for %s failed: %s", mid, e)
            output_embeddings[mid] = []

    # Step 4: Run evaluation pipeline
    custom_weights = request.weights

    BUSINESS_TO_INTERNAL = {
        "answer_accuracy": "cross_encoder",
        "semantic_similarity": "embedding_similarity",
        "response_efficiency": "response_efficiency",
        # Legacy frontend mapping compatibility
        "accuracy": "cross_encoder",
        "relevance": "embedding_similarity",
        "conciseness": "response_efficiency",
    }

    if custom_weights:
        resolved_weights = {}
        for biz_key, value in custom_weights.items():
            internal_key = BUSINESS_TO_INTERNAL.get(biz_key, biz_key)
            if internal_key in ["cross_encoder", "embedding_similarity", "response_efficiency"]:
                resolved_weights[internal_key] = value
        custom_weights = resolved_weights

    sigma = request.sigma if request.sigma is not None else RESPONSE_EFFICIENCY_CONFIG["default_sigma"]

    try:
        eval_result = evaluation.evaluate_single(
            prompt=user_prompt,
            ground_truth=ground_truth,
            model_outputs=model_outputs,
            model_token_usage=model_token_usage,
            gt_embedding=gt_embedding,
            output_embeddings=output_embeddings,
            ce_model=ce_model,
            sigma=sigma,
            weights=custom_weights,
        )
    except Exception as e:
        logger.exception("Evaluation pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation pipeline error: {e}",
        )

    # Step 5: Build response
    results: list[ModelResult] = []
    winner_result: ModelResult | None = None

    for r in eval_result["results"]:
        provider, model_name = _parse_model_id(r["model_id"])
        mr = ModelResult(
            model_id=r["model_id"],
            model_name=model_name,
            provider=provider,
            output=r["output"],
            cross_encoder=r["cross_encoder"],
            embedding_similarity=r["embedding_similarity"],
            response_efficiency=r["response_efficiency"],
            input_tokens=r.get("input_tokens", 0),
            output_tokens=r.get("output_tokens", 0),
            input_cost=r.get("input_cost", 0.0),
            output_cost=r.get("output_cost", 0.0),
            total_cost=r.get("total_cost", 0.0),
            composite=r["composite"],
        )
        results.append(mr)

        if r["model_id"] == eval_result["winner"]:
            winner_result = mr

    if winner_result is None and results:
        winner_result = max(results, key=lambda r: r.composite)

    return EvaluateResponse(
        user_prompt=user_prompt,
        ground_truth=ground_truth,
        winner=winner_result,
        results=results,
    )


# ────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "region": REGION}

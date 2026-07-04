const API_BASE = '/api';

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function evaluate({ userPrompt, systemPrompt, outputParser, outputSchema, groundTruth, modelIds, weights, sigma }) {
  const body = {
    user_prompt: userPrompt,
    system_prompt: systemPrompt?.trim() || '',
    ground_truth: groundTruth,
    model_ids: modelIds,
    output_schema: '',
  };
  const schemaOrParser = outputSchema?.trim() || outputParser?.trim();
  if (schemaOrParser) {
    body.output_schema = schemaOrParser;
    body.output_parser = schemaOrParser;
  }
  if (weights) body.weights = weights;
  if (sigma != null) body.sigma = sigma;

  const res = await fetch(`${API_BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Evaluation failed' }));
    throw new Error(err.detail || 'Evaluation failed');
  }
  return res.json();
}

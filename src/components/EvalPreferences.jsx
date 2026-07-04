import { useState, useCallback } from 'react';

const CRITERIA = [
  {
    key: 'answer_accuracy',
    label: 'Answer Accuracy',
    description: 'Measures how accurately the generated response matches the meaning of the expected answer (CrossEncoder).',
    defaultPct: 45,
  },
  {
    key: 'semantic_similarity',
    label: 'Semantic Similarity',
    description: 'Measures how closely the generated response matches the overall meaning of the expected answer (Titan Embeddings V2).',
    defaultPct: 30,
  },
  {
    key: 'response_efficiency',
    label: 'Response Efficiency',
    description: 'Measures whether the generated response is appropriately sized compared to the expected answer (Ground Truth token count).',
    defaultPct: 15,
  },
];

const PRESETS = {
  default: { answer_accuracy: 0.45, semantic_similarity: 0.30, response_efficiency: 0.15 },
  accuracy_first: { answer_accuracy: 0.60, semantic_similarity: 0.20, response_efficiency: 0.10 },
  semantic_focus: { answer_accuracy: 0.30, semantic_similarity: 0.50, response_efficiency: 0.10 },
  balanced: { answer_accuracy: 0.33, semantic_similarity: 0.33, response_efficiency: 0.34 },
};

export default function EvalPreferences({ weights, setWeights, sigma = 0.5, setSigma, disabled }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSliderChange = useCallback((changedKey, newVal) => {
    if (disabled) return;
    setWeights((prev) => {
      const updated = { ...prev, [changedKey]: newVal };
      return updated;
    });
  }, [disabled, setWeights]);

  const applyPreset = useCallback((presetKey) => {
    if (disabled) return;
    setWeights({ ...PRESETS[presetKey] });
  }, [disabled, setWeights]);

  const totalRawWeight = (weights.answer_accuracy || 0) + (weights.semantic_similarity || 0) + (weights.response_efficiency || 0);

  return (
    <section className="glass-card eval-preferences-section fade-in" id="eval-preferences-section">
      <div className="section-header">
        <h2 className="section-title">
          <span className="section-icon">⚖</span>
          Evaluation Criteria & Weights
        </h2>
        <div className="preset-buttons">
          {Object.keys(PRESETS).map((key) => (
            <button
              key={key}
              type="button"
              className="preset-btn"
              onClick={() => applyPreset(key)}
              disabled={disabled}
            >
              {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </button>
          ))}
        </div>
      </div>

      <div className="simplified-weights-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', margin: '16px 0' }}>
        {CRITERIA.map((c) => {
          const rawVal = weights[c.key] ?? (c.defaultPct / 100);
          const normalizedPct = totalRawWeight > 0 ? Math.round((rawVal / totalRawWeight) * 100) : c.defaultPct;

          return (
            <div
              key={c.key}
              className="weight-card"
              style={{
                background: 'rgba(15, 23, 42, 0.45)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '10px',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong style={{ fontSize: '15px', color: '#f8fafc' }}>{c.label}</strong>
                <span style={{
                  background: 'rgba(59, 130, 246, 0.15)',
                  color: '#3b82f6',
                  fontWeight: '700',
                  padding: '4px 8px',
                  borderRadius: '6px',
                  fontSize: '13px'
                }}>
                  {normalizedPct}% Weight
                </span>
              </div>
              <p style={{ fontSize: '12px', color: '#94a3b8', margin: 0, lineHeight: '1.4' }}>
                {c.description}
              </p>
              <div style={{ marginTop: 'auto', paddingTop: '8px' }}>
                <input
                  type="range"
                  min="0.05"
                  max="0.80"
                  step="0.05"
                  disabled={disabled}
                  value={rawVal}
                  onChange={(e) => handleSliderChange(c.key, parseFloat(e.target.value))}
                  style={{ width: '100%', cursor: disabled ? 'not-allowed' : 'pointer' }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="cost-info-banner" style={{
        margin: '16px 0',
        padding: '14px 18px',
        background: 'rgba(16, 185, 129, 0.08)',
        border: '1px solid rgba(16, 185, 129, 0.25)',
        borderRadius: '10px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <span style={{ fontSize: '20px' }}>💲</span>
        <div>
          <strong style={{ color: '#10b981', fontSize: '14px', display: 'block' }}>
            Estimated Cost (Informational Metric)
          </strong>
          <span style={{ color: '#cbd5e1', fontSize: '12px' }}>
            Inference cost is calculated automatically for every model based on official Bedrock pricing and token usage. It is displayed separately to aid cost-efficiency decisions and does NOT affect the composite evaluation score.
          </span>
        </div>
      </div>

      <div className="advanced-settings-toggle" style={{ marginTop: '16px', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '12px' }}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            background: 'none',
            border: 'none',
            color: '#64748b',
            fontSize: '12px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: 0
          }}
        >
          <span>{showAdvanced ? '▾' : '▸'} Advanced Tolerance Settings (Response Efficiency)</span>
        </button>

        {showAdvanced && (
          <div className="advanced-settings-box" style={{ marginTop: '10px', padding: '12px', background: 'rgba(15, 23, 42, 0.6)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <label htmlFor="custom-tolerance-input" style={{ fontSize: '12px', color: '#cbd5e1' }}>
              Response Length Tolerance Value (Sigma):
            </label>
            <input
              id="custom-tolerance-input"
              type="number"
              step="0.05"
              min="0.1"
              max="2.0"
              disabled={disabled}
              value={sigma}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val) && val > 0) setSigma && setSigma(val);
              }}
              style={{
                width: '100px',
                padding: '4px 8px',
                borderRadius: '6px',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                background: 'rgba(0, 0, 0, 0.3)',
                color: '#ffffff',
                fontSize: '13px'
              }}
            />
          </div>
        )}
      </div>
    </section>
  );
}

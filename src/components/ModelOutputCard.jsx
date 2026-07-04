export default function ModelOutputCard({ result, isWinner }) {
  if (!result) return null;

  const isError = result.output?.startsWith('__ERROR__');

  const formatPercentage = (val) => {
    if (val === undefined || val === null) return '0%';
    return `${Math.round(val * 100)}%`;
  };

  const formatCost = (val) => {
    if (val === undefined || val === null) return '$0.0000';
    return `$${Number(val).toFixed(4)}`;
  };

  const formatResponseContent = (output) => {
    if (!output && output !== '') return '(No output generated)';
    if (typeof output !== 'string') {
      try {
        return JSON.stringify(output, null, 2);
      } catch {
        return String(output);
      }
    }
    if (output.startsWith('__ERROR__')) return output;

    const trimmed = output.trim();
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        const parsed = JSON.parse(trimmed);
        return JSON.stringify(parsed, null, 2);
      } catch {
        // Not valid JSON or parsing failed, preserve original text exactly
      }
    }
    return output;
  };

  const formattedOutput = formatResponseContent(result.output);

  return (
    <article
      className={`glass-card model-output-card fade-in ${isWinner ? 'winner-border' : ''}`}
      id={`model-output-card-${result.model_id}`}
    >
      <div className="card-header">
        <div className="card-header-left">
          {isWinner && <span className="trophy-badge" title="Top Performer">🏆</span>}
          <h3 className="card-model-name">{result.model_name || result.model_id}</h3>
        </div>
        <div className="card-header-right">
          <span className="provider-tag">{result.provider || 'Bedrock'}</span>
          <span className="composite-badge" title="Overall Composite Score">
            Overall: {formatPercentage(result.composite)}
          </span>
        </div>
      </div>

      <div className="card-metrics-row" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', margin: '12px 0' }}>
        <div className="card-metric badge-blue" style={{ flex: 1, minWidth: '130px', padding: '8px 12px', borderRadius: '8px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
          <span className="cm-label" style={{ display: 'block', fontSize: '11px', color: '#94a3b8' }}>Answer Accuracy</span>
          <span className="cm-value" style={{ fontSize: '16px', fontWeight: 700, color: '#60a5fa' }}>{formatPercentage(result.cross_encoder)}</span>
        </div>

        <div className="card-metric badge-purple" style={{ flex: 1, minWidth: '130px', padding: '8px 12px', borderRadius: '8px', background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
          <span className="cm-label" style={{ display: 'block', fontSize: '11px', color: '#94a3b8' }}>Semantic Similarity</span>
          <span className="cm-value" style={{ fontSize: '16px', fontWeight: 700, color: '#a78bfa' }}>{formatPercentage(result.embedding_similarity)}</span>
        </div>

        <div className="card-metric badge-amber" style={{ flex: 1, minWidth: '130px', padding: '8px 12px', borderRadius: '8px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
          <span className="cm-label" style={{ display: 'block', fontSize: '11px', color: '#94a3b8' }}>Response Efficiency</span>
          <span className="cm-value" style={{ fontSize: '16px', fontWeight: 700, color: '#fbbf24' }}>{formatPercentage(result.response_efficiency)}</span>
        </div>

        <div className="card-metric badge-green" style={{ flex: 1, minWidth: '130px', padding: '8px 12px', borderRadius: '8px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
          <span className="cm-label" style={{ display: 'block', fontSize: '11px', color: '#94a3b8' }}>Estimated Cost</span>
          <span className="cm-value" style={{ fontSize: '16px', fontWeight: 700, color: '#34d399' }}>{formatCost(result.total_cost)}</span>
        </div>
      </div>

      <div className="card-body">
        <div className="output-section">
          <h4 className="output-heading">Generated Response</h4>
          <pre className={`output-text ${isError ? 'error-text' : ''}`}>
            {formattedOutput}
          </pre>
        </div>
      </div>

      <div className="card-footer" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#64748b', marginTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '8px' }}>
        <span>Tokens: {result.input_tokens || 0} in / {result.output_tokens || 0} out</span>
        <span>ID: {result.model_id}</span>
      </div>
    </article>
  );
}

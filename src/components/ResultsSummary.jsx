import { useState, useEffect } from 'react';

export default function ResultsSummary({ results }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  const winner = results?.winner;
  const winnerResult = results?.results?.find(
    (r) => r.model_id === winner?.model_id
  );

  useEffect(() => {
    if (!winner?.composite) return;
    const target = winner.composite;
    let start = 0;
    const duration = 1200;
    const startTime = performance.now();

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      start = target * eased;
      setAnimatedScore(start);
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [winner]);

  if (!winner || !winnerResult) return null;

  const metrics = {
    cross_encoder: winnerResult.cross_encoder,
    embedding_similarity: winnerResult.embedding_similarity,
    response_efficiency: winnerResult.response_efficiency,
  };

  const metricLabels = {
    cross_encoder: 'Answer Accuracy',
    embedding_similarity: 'Semantic Similarity',
    response_efficiency: 'Response Efficiency',
  };

  const formatPercentage = (val) => {
    if (val === undefined || val === null) return '0%';
    return `${Math.round(val * 100)}%`;
  };

  const formatCost = (val) => {
    if (val === undefined || val === null) return '$0.0000';
    return `$${Number(val).toFixed(4)}`;
  };

  return (
    <section className="results-summary fade-in" id="results-summary-section">
      <div className="winner-card">
        <div className="winner-glow" />
        <div className="winner-content">
          <div className="winner-trophy">🏆</div>
          <div className="winner-label">Top Performer</div>
          <h2 className="winner-model-name" id="winner-model-name">
            {winnerResult.model_name || winner.model_id}
          </h2>
          <div className="winner-score-container">
            <div className="winner-score" id="winner-composite-score">
              {formatPercentage(animatedScore)}
            </div>
            <div className="winner-score-label">Overall Score</div>
          </div>

          <div className="winner-metrics">
            {Object.entries(metricLabels).map(([key, label]) => {
              const value = metrics[key] ?? 0;
              return (
                <div className="winner-metric" key={key} id={`winner-metric-${key}`}>
                  <div className="winner-metric-header">
                    <span className="winner-metric-label">{label}</span>
                    <span className="winner-metric-value">{formatPercentage(value)}</span>
                  </div>
                  <div className="winner-metric-bar-bg">
                    <div
                      className="winner-metric-bar-fill"
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}

            <div className="winner-metric" id="winner-metric-cost" style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px dashed rgba(255, 255, 255, 0.1)' }}>
              <div className="winner-metric-header">
                <span className="winner-metric-label" style={{ color: '#34d399' }}>Estimated Cost</span>
                <span className="winner-metric-value" style={{ color: '#34d399' }}>{formatCost(winnerResult.total_cost)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

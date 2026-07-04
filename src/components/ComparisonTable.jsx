import { useState } from 'react';

export default function ComparisonTable({ results }) {
  const [sortField, setSortField] = useState('composite');
  const [sortAsc, setSortAsc] = useState(false);

  if (!results?.results?.length) return null;

  const winnerId = results.winner?.model_id;

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedResults = [...results.results].sort((a, b) => {
    let valA = a[sortField] ?? 0;
    let valB = b[sortField] ?? 0;
    if (typeof valA === 'string') {
      return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return sortAsc ? valA - valB : valB - valA;
  });

  const formatPercentage = (val) => {
    if (val === undefined || val === null) return '0%';
    return `${Math.round(val * 100)}%`;
  };

  const formatCost = (val) => {
    if (val === undefined || val === null) return '$0.0000';
    return `$${Number(val).toFixed(4)}`;
  };

  const columns = [
    { key: 'model_id', label: 'Model', align: 'left' },
    { key: 'cross_encoder', label: 'Answer Accuracy', align: 'center' },
    { key: 'embedding_similarity', label: 'Semantic Similarity', align: 'center' },
    { key: 'response_efficiency', label: 'Response Efficiency', align: 'center' },
    { key: 'total_cost', label: 'Estimated Cost', align: 'center' },
    { key: 'composite', label: 'Overall Score', align: 'center' },
  ];

  return (
    <section className="glass-card comparison-table-section fade-in" id="comparison-table-section">
      <div className="section-header">
        <h2 className="section-title">
          <span className="section-icon">▤</span>
          Model Comparison Table
        </h2>
      </div>

      <div className="table-responsive">
        <table className="comparison-table" id="comparison-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`th-sortable ${sortField === col.key ? 'active' : ''}`}
                  style={{ textAlign: col.align }}
                  onClick={() => handleSort(col.key)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: col.align === 'center' ? 'center' : 'flex-start', gap: '6px' }}>
                    <span>{col.label}</span>
                    {sortField === col.key && (
                      <span className="sort-arrow">{sortAsc ? '▲' : '▼'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((r) => {
              const isWinner = r.model_id === winnerId;
              return (
                <tr
                  key={r.model_id}
                  className={`comparison-row ${isWinner ? 'winner-row' : ''}`}
                  id={`comparison-row-${r.model_id}`}
                >
                  <td className="td-model">
                    <div className="model-cell">
                      {isWinner && <span className="row-trophy" title="Top Performer">🏆</span>}
                      <div>
                        <div className="model-cell-name">{r.model_name || r.model_id}</div>
                        <div className="model-cell-id">{r.model_id}</div>
                      </div>
                    </div>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className="metric-badge badge-blue">
                      {formatPercentage(r.cross_encoder)}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className="metric-badge badge-purple">
                      {formatPercentage(r.embedding_similarity)}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className="metric-badge badge-amber">
                      {formatPercentage(r.response_efficiency)}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className="metric-badge badge-green" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', fontWeight: '600' }}>
                      {formatCost(r.total_cost)}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <div className="overall-score-badge">
                      {formatPercentage(r.composite)}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

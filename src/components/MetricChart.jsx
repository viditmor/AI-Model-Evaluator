import { useState, useMemo } from 'react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444', '#06b6d4'];

const METRIC_LABELS = {
  cross_encoder: 'Answer Accuracy',
  embedding_similarity: 'Semantic Similarity',
  response_efficiency: 'Response Efficiency',
};

const METRIC_KEYS = Object.keys(METRIC_LABELS);

export default function MetricChart({ results }) {
  const [chartType, setChartType] = useState('radar');

  const modelNames = useMemo(() => {
    return results?.results?.map((r) => r.model_id) || [];
  }, [results]);

  const radarData = useMemo(() => {
    if (!results?.results) return [];
    return METRIC_KEYS.map((key) => {
      const entry = { metric: METRIC_LABELS[key] };
      results.results.forEach((r) => {
        entry[r.model_id] = r[key] ?? 0;
      });
      return entry;
    });
  }, [results]);

  const barData = useMemo(() => {
    if (!results?.results) return [];
    return results.results.map((r) => {
      const entry = { model: r.model_name || r.model_id };
      METRIC_KEYS.forEach((key) => {
        entry[METRIC_LABELS[key]] = r[key] ?? 0;
      });
      return entry;
    });
  }, [results]);

  if (!results?.results?.length) return null;

  return (
    <section className="glass-card metric-chart-section fade-in" id="metric-chart-section">
      <div className="section-header">
        <h2 className="section-title">
          <span className="section-icon">◉</span>
          Quality Metrics Comparison
        </h2>
        <div className="chart-toggle">
          <button
            className={`chart-toggle-btn ${chartType === 'radar' ? 'active' : ''}`}
            id="chart-toggle-radar"
            onClick={() => setChartType('radar')}
          >
            Radar
          </button>
          <button
            className={`chart-toggle-btn ${chartType === 'bar' ? 'active' : ''}`}
            id="chart-toggle-bar"
            onClick={() => setChartType('bar')}
          >
            Bar
          </button>
        </div>
      </div>

      <div className="chart-container" id="chart-container">
        {chartType === 'radar' ? (
          <ResponsiveContainer width="100%" height={380}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fill: '#e2e8f0', fontSize: 13, fontWeight: 600 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 1]}
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickCount={5}
              />
              {modelNames.map((name, i) => (
                <Radar
                  key={name}
                  name={name}
                  dataKey={name}
                  stroke={COLORS[i % COLORS.length]}
                  fill={COLORS[i % COLORS.length]}
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              ))}
              <Legend
                wrapperStyle={{ color: '#94a3b8', fontSize: 12, paddingTop: 16 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontSize: 12,
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="model"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              {Object.values(METRIC_LABELS).map((label, i) => (
                <Bar
                  key={label}
                  dataKey={label}
                  fill={COLORS[i % COLORS.length]}
                  radius={[4, 4, 0, 0]}
                  opacity={0.85}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

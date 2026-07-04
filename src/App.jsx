import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchModels, evaluate } from './api/client';
import Header from './components/Header';
import PromptInput from './components/PromptInput';
import ModelSelector from './components/ModelSelector';
import EvalPreferences from './components/EvalPreferences';
import EvaluateButton from './components/EvaluateButton';
import ResultsSummary from './components/ResultsSummary';
import MetricChart from './components/MetricChart';
import ComparisonTable from './components/ComparisonTable';
import ModelOutputCard from './components/ModelOutputCard';

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [outputParser, setOutputParser] = useState('');
  const [groundTruth, setGroundTruth] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [weights, setWeights] = useState({
    answer_accuracy: 0.45,
    semantic_similarity: 0.30,
    response_efficiency: 0.15,
  });
  const [sigma, setSigma] = useState(0.5);

  const resultsRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setModelsLoading(true);
    fetchModels()
      .then((data) => {
        if (!cancelled) {
          const modelList = Array.isArray(data) ? data : data?.models || [];
          setModels(modelList);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError('Failed to load models. Is the API server running?');
          console.error('Failed to fetch models:', err);
        }
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const handleEvaluate = useCallback(async () => {
    if (!prompt.trim() || !groundTruth.trim() || selectedModels.length === 0) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await evaluate({
        userPrompt: prompt,
        systemPrompt,
        outputParser,
        groundTruth,
        modelIds: selectedModels,
        weights,
        sigma,
      });
      setResults(data);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      setError(err.message || 'Evaluation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [prompt, systemPrompt, outputParser, groundTruth, selectedModels, weights, sigma]);

  const dismissError = useCallback(() => setError(null), []);

  const isEvaluateDisabled =
    !prompt.trim() || !groundTruth.trim() || selectedModels.length === 0;

  const winnerId = results?.winner?.model_id;

  return (
    <div className="app" id="app-root">
      <div className="app-bg" />

      {error && (
        <div className="error-toast" id="error-toast" role="alert">
          <div className="error-toast-content">
            <span className="error-toast-icon">⚠</span>
            <span className="error-toast-message">{error}</span>
            <button
              className="error-toast-close"
              id="error-toast-close"
              onClick={dismissError}
              aria-label="Dismiss error"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <div className="app-container">
        <Header />

        <main className="app-main" id="app-main">
          <PromptInput
            prompt={prompt}
            setPrompt={setPrompt}
            systemPrompt={systemPrompt}
            setSystemPrompt={setSystemPrompt}
            outputParser={outputParser}
            setOutputParser={setOutputParser}
            groundTruth={groundTruth}
            setGroundTruth={setGroundTruth}
            disabled={loading}
          />

          {modelsLoading ? (
            <div className="glass-card loading-card">
              <div className="spinner" />
              <p className="loading-text">Loading available models...</p>
            </div>
          ) : (
            <ModelSelector
              models={models}
              selectedModels={selectedModels}
              setSelectedModels={setSelectedModels}
              disabled={loading}
            />
          )}

          <EvalPreferences
            weights={weights}
            setWeights={setWeights}
            sigma={sigma}
            setSigma={setSigma}
            disabled={loading}
          />

          <EvaluateButton
            onClick={handleEvaluate}
            loading={loading}
            disabled={isEvaluateDisabled}
          />

          {results && (
            <div className="results-container" ref={resultsRef} id="results-container">
              <ResultsSummary results={results} />
              <MetricChart results={results} />
              <ComparisonTable results={results} />

              <section className="model-outputs-section" id="model-outputs-section">
                <div className="section-header-standalone">
                  <h2 className="section-title">
                    <span className="section-icon">◫</span>
                    Model Outputs
                  </h2>
                </div>
                <div className="model-outputs-grid">
                  {results.results?.map((r) => (
                    <ModelOutputCard
                      key={r.model_id}
                      result={r}
                      isWinner={r.model_id === winnerId}
                    />
                  ))}
                </div>
              </section>
            </div>
          )}
        </main>

        <footer className="app-footer" id="app-footer">
          <p>LLM Evaluation Platform · Multi-Model Comparison Dashboard</p>
        </footer>
      </div>
    </div>
  );
}

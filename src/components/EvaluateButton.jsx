export default function EvaluateButton({ onClick, loading, disabled }) {
  return (
    <div className="evaluate-button-wrapper" id="evaluate-button-wrapper">
      <button
        className={`evaluate-button ${loading ? 'loading' : ''}`}
        id="evaluate-button"
        onClick={onClick}
        disabled={disabled || loading}
        aria-busy={loading}
      >
        {loading ? (
          <span className="evaluate-button-content">
            <span className="spinner" aria-hidden="true" />
            <span>Evaluating...</span>
          </span>
        ) : (
          <span className="evaluate-button-content">
            <span className="evaluate-icon" aria-hidden="true">⚡</span>
            <span>Run Evaluation</span>
          </span>
        )}
      </button>
      {disabled && !loading && (
        <p className="evaluate-hint" id="evaluate-hint">
          Enter a prompt, ground truth, and select at least one model to begin.
        </p>
      )}
    </div>
  );
}

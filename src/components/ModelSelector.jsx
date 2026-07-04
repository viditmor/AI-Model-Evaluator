import { useState, useMemo } from 'react';

export default function ModelSelector({ models, selectedModels, setSelectedModels, disabled }) {
  const [collapsedProviders, setCollapsedProviders] = useState({});

  const grouped = useMemo(() => {
    if (!models || !Array.isArray(models)) return {};
    return models.reduce((acc, model) => {
      const provider = model.provider || 'Unknown';
      if (!acc[provider]) acc[provider] = [];
      acc[provider].push(model);
      return acc;
    }, {});
  }, [models]);

  const allModelIds = useMemo(() => {
    return models?.map((m) => m.model_id) || [];
  }, [models]);

  const totalSelected = selectedModels.length;

  const toggleModel = (modelId) => {
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((id) => id !== modelId) : [...prev, modelId]
    );
  };

  const toggleProvider = (provider) => {
    setCollapsedProviders((prev) => ({ ...prev, [provider]: !prev[provider] }));
  };

  const selectAllProvider = (provider) => {
    const providerIds = grouped[provider].map((m) => m.model_id);
    setSelectedModels((prev) => [...new Set([...prev, ...providerIds])]);
  };

  const deselectAllProvider = (provider) => {
    const providerIds = new Set(grouped[provider].map((m) => m.model_id));
    setSelectedModels((prev) => prev.filter((id) => !providerIds.has(id)));
  };

  const selectAll = () => setSelectedModels([...allModelIds]);
  const clearAll = () => setSelectedModels([]);

  const isProviderFullySelected = (provider) => {
    return grouped[provider].every((m) => selectedModels.includes(m.model_id));
  };

  const isProviderPartiallySelected = (provider) => {
    const ids = grouped[provider].map((m) => m.model_id);
    const selectedCount = ids.filter((id) => selectedModels.includes(id)).length;
    return selectedCount > 0 && selectedCount < ids.length;
  };

  if (!models || models.length === 0) {
    return (
      <section className="glass-card model-selector-section" id="model-selector-section">
        <div className="section-header">
          <h2 className="section-title">
            <span className="section-icon">◈</span>
            Model Selection
          </h2>
        </div>
        <div className="model-selector-empty">
          <div className="empty-icon">⊘</div>
          <p>No models available. Make sure the API server is running.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="glass-card model-selector-section" id="model-selector-section">
      <div className="section-header">
        <h2 className="section-title">
          <span className="section-icon">◈</span>
          Model Selection
        </h2>
        <div className="global-actions">
          <button
            className="action-btn select-all-btn"
            id="global-select-all-btn"
            onClick={selectAll}
            disabled={disabled}
          >
            Select All
          </button>
          <button
            className="action-btn clear-all-btn"
            id="global-clear-all-btn"
            onClick={clearAll}
            disabled={disabled}
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="provider-groups">
        {Object.entries(grouped).map(([provider, providerModels]) => {
          const isCollapsed = collapsedProviders[provider];
          const fullySelected = isProviderFullySelected(provider);
          const partiallySelected = isProviderPartiallySelected(provider);

          return (
            <div className="provider-group" key={provider} id={`provider-${provider.toLowerCase().replace(/\s+/g, '-')}`}>
              <div
                className="provider-header"
                onClick={() => toggleProvider(provider)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && toggleProvider(provider)}
              >
                <div className="provider-header-left">
                  <span className={`provider-chevron ${isCollapsed ? '' : 'expanded'}`}>›</span>
                  <span className="provider-name">{provider}</span>
                  <span className="provider-count-badge">{providerModels.length}</span>
                </div>
                <div className="provider-header-right">
                  {partiallySelected && <span className="partial-badge">partial</span>}
                  <button
                    className="action-btn-sm"
                    id={`provider-toggle-${provider.toLowerCase().replace(/\s+/g, '-')}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      fullySelected ? deselectAllProvider(provider) : selectAllProvider(provider);
                    }}
                    disabled={disabled}
                  >
                    {fullySelected ? 'Deselect All' : 'Select All'}
                  </button>
                </div>
              </div>

              {!isCollapsed && (
                <div className="provider-models">
                  {providerModels.map((model) => {
                    const isSelected = selectedModels.includes(model.model_id);
                    return (
                      <label
                        className={`model-checkbox-label ${isSelected ? 'selected' : ''}`}
                        key={model.model_id}
                        id={`model-label-${model.model_id}`}
                      >
                        <div className={`custom-checkbox ${isSelected ? 'checked' : ''}`}>
                          <svg className="check-icon" viewBox="0 0 12 10" fill="none">
                            <path
                              d="M1 5L4.5 8.5L11 1.5"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        </div>
                        <input
                          type="checkbox"
                          className="sr-only"
                          checked={isSelected}
                          onChange={() => toggleModel(model.model_id)}
                          disabled={disabled}
                          id={`model-checkbox-${model.model_id}`}
                        />
                        <span className="model-name">{model.model_name || model.model_id}</span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="model-selector-footer">
        <span className="selected-count" id="selected-model-count">
          <span className="count-number">{totalSelected}</span> model{totalSelected !== 1 ? 's' : ''} selected
        </span>
      </div>
    </section>
  );
}

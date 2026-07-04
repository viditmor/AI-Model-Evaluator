import { useRef, useEffect, useCallback } from 'react';

export default function PromptInput({
  prompt,
  setPrompt,
  systemPrompt,
  setSystemPrompt,
  outputParser,
  setOutputParser,
  groundTruth,
  setGroundTruth,
  disabled,
}) {
  const promptRef = useRef(null);
  const systemRef = useRef(null);
  const outputRef = useRef(null);
  const groundTruthRef = useRef(null);

  const autoResize = useCallback((textarea, isLarge = false) => {
    if (!textarea) return;
    textarea.style.height = 'auto';
    const minH = isLarge ? 260 : 110;
    const maxH = isLarge ? 650 : 350;
    const newHeight = Math.min(Math.max(textarea.scrollHeight, minH), maxH);
    textarea.style.height = `${newHeight}px`;
  }, []);

  useEffect(() => { autoResize(promptRef.current, true); }, [prompt, autoResize]);
  useEffect(() => { autoResize(systemRef.current, false); }, [systemPrompt, autoResize]);
  useEffect(() => { autoResize(outputRef.current, false); }, [outputParser, autoResize]);
  useEffect(() => { autoResize(groundTruthRef.current, true); }, [groundTruth, autoResize]);

  return (
    <div className="prompt-layout-container" id="prompt-input-section">
      {/* ── LEFT PANEL: Model Input (~65% width on desktop) ──────── */}
      <section className="glass-card panel-model-input" id="panel-model-input">
        <div className="panel-header">
          <h2 className="section-title">
            <span className="section-icon">⚡</span>
            Model Input
          </h2>
          <p className="panel-description">
            These inputs are sent directly to the selected language models.
          </p>
        </div>

        <div className="panel-fields-stack">
          {/* 1. User Prompt (Required) */}
          <div className="inner-field-card">
            <label className="input-label" htmlFor="prompt-textarea">
              <span className="input-label-dot prompt-dot" />
              User Prompt
              <span className="required-badge">Required</span>
            </label>
            <textarea
              ref={promptRef}
              id="prompt-textarea"
              className="text-input large-textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter the task or question you want the models to answer."
              disabled={disabled}
              rows={12}
            />
            <div className="input-meta">
              <span className="char-count" id="prompt-char-count">
                {prompt.length} characters
              </span>
            </div>
          </div>

          {/* 2. System Prompt (Optional) */}
          <div className="inner-field-card">
            <label className="input-label" htmlFor="system-prompt-textarea">
              <span className="input-label-dot system-dot" />
              System Prompt
              <span className="optional-badge">Optional</span>
            </label>
            <textarea
              ref={systemRef}
              id="system-prompt-textarea"
              className="text-input medium-textarea"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Optional instructions describing how the model should behave."
              disabled={disabled}
              rows={4}
            />
            <div className="input-meta">
              <span className="char-count" id="system-prompt-char-count">
                {systemPrompt.length} characters
              </span>
            </div>
          </div>

          {/* 3. Output Schema (Optional) */}
          <div className="inner-field-card">
            <label className="input-label" htmlFor="output-parser-textarea">
              <span className="input-label-dot format-dot" />
              Output Schema
              <span className="optional-badge">Optional</span>
            </label>
            <p className="field-subtext">Define the expected structure of the model response using JSON Schema.</p>
            <textarea
              ref={outputRef}
              id="output-parser-textarea"
              className="text-input medium-textarea schema-textarea"
              value={outputParser}
              onChange={(e) => setOutputParser(e.target.value)}
              placeholder={`{\n  "type": "object",\n  "properties": {\n    "answer": {\n      "type": "string"\n    },\n    "confidence": {\n      "type": "number"\n    }\n  },\n  "required": [\n    "answer",\n    "confidence"\n  ]\n}`}
              disabled={disabled}
              rows={6}
            />
            <div className="input-meta">
              <span className="char-count" id="output-parser-char-count">
                {outputParser.length} characters
              </span>
            </div>
          </div>
        </div>

        <div className="panel-footer-note model-note">
          <span className="note-check">✓</span> Sent to every selected model.
        </div>
      </section>

      {/* ── RIGHT PANEL: Evaluation Reference (~35% width on desktop) ── */}
      <section className="glass-card panel-eval-reference" id="panel-eval-reference">
        <div className="panel-header">
          <h2 className="section-title">
            <span className="section-icon eval-icon">⚖️</span>
            Evaluation Reference
          </h2>
          <p className="panel-description">
            This information is NOT sent to the model. It is used only to evaluate the generated responses.
          </p>
        </div>

        <div className="panel-fields-stack">
          <div className="inner-field-card eval-inner-card">
            <label className="input-label" htmlFor="ground-truth-textarea">
              <span className="input-label-dot truth-dot" />
              Ground Truth
              <span className="required-badge eval-required-badge">Required</span>
            </label>
            <textarea
              ref={groundTruthRef}
              id="ground-truth-textarea"
              className="text-input large-textarea eval-textarea"
              value={groundTruth}
              onChange={(e) => setGroundTruth(e.target.value)}
              placeholder="Enter the expected or ideal answer."
              disabled={disabled}
              rows={12}
            />
            <div className="input-meta">
              <span className="char-count" id="ground-truth-char-count">
                {groundTruth.length} characters
              </span>
            </div>
          </div>
        </div>

        <div className="panel-footer-note eval-note">
          <span className="note-check">✓</span> Used only for evaluation and scoring.
        </div>
      </section>
    </div>
  );
}

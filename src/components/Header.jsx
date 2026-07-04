import { useState, useEffect } from 'react';

export default function Header() {
  const [glowIntensity, setGlowIntensity] = useState(0);

  useEffect(() => {
    let frame;
    let start = null;
    const animate = (timestamp) => {
      if (!start) start = timestamp;
      const elapsed = (timestamp - start) / 1000;
      setGlowIntensity(Math.sin(elapsed * 1.5) * 0.5 + 0.5);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <header className="header" id="app-header">
      <div className="header-glow" style={{ opacity: glowIntensity * 0.15 }} />
      <div className="header-content">
        <div className="header-badge">
          <span className="header-badge-dot" />
          <span className="header-badge-text">AI-Powered Analysis</span>
        </div>
        <h1 className="header-title" id="header-title">
          <span className="header-title-gradient">LLM Evaluation</span>
          <span className="header-title-white"> Platform</span>
        </h1>
        <p className="header-subtitle" id="header-subtitle">
          Multi-Model Comparison Engine
        </p>
        <div className="header-divider" />
      </div>
    </header>
  );
}

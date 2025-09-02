'use client';

import { useState } from 'react';

type Metrics = { structure: number; plan_adherence: number; hallucination_proxy: number };

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';

const DEMOS = [
  'Write a policy brief on urban heat mitigation for a midsize US city. Use sections: Introduction, Methods/Literature, Findings, Policy Implications, Conclusion.',
  'Design a 5-day lesson plan to teach recursion to high schoolers with Objectives, Background, Activities (day-by-day), Assessment, Extensions.',
  'Draft a research-methods section for a usability study on note-taking apps with Participants, Materials, Procedure, Measures, Analysis Plan, Ethics.',
  'Create a troubleshooting guide for a failing CI pipeline with Symptoms, Quick Checks, Logs, Common Root Causes, Fix Steps, Prevention.',
  'Outline and write a Specific Aims page for a wildfire forecasting grant (Problem, Aim 1, Aim 2, Aim 3, Expected Impact).',
];

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [outline, setOutline] = useState<string[]>([]);
  const [baseline, setBaseline] = useState('');
  const [planText, setPlanText] = useState('');
  const [mb, setMB] = useState<Metrics | null>(null);
  const [mp, setMP] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onCompare() {
    setErr(null);
    if (!prompt.trim()) {
      setErr('Please enter a prompt first.');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/v1/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`API error ${res.status}: ${t}`);
      }
      const data = await res.json();
      setOutline(data.outline || []);
      setBaseline(data.baseline_text || '');
      setPlanText(data.plan_text || '');
      setMB(data.metrics_baseline || null);
      setMP(data.metrics_plan || null);
    } catch (e: unknown) {
      if (e instanceof Error) {
        setErr(e.message);
      } else {
        setErr('Request failed');
      }
    } finally {
      setLoading(false);
    }
  }

  function pasteDemo(i: number) {
    setPrompt(DEMOS[i]);
    setOutline([]); setBaseline(''); setPlanText(''); setMB(null); setMP(null); setErr(null);
  }

  function fmt(n?: number) {
    if (n == null) return '—';
    return n.toFixed(3);
  }

  return (
    <div className="container">
      <div className="header">
        <h1>Struct-Align</h1>
        <span className="badge">Plan-First → Write</span>
      </div>
      <p className="subtle">
        Fine-tuned with a <b>structure-aware reward</b> so long-form answers are organized and follow the plan.
      </p>

      {/* Prompt card */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="small">Backend: <code>{API_BASE}/v1/compare</code></div>
          {err && <div className="small" style={{ color: 'var(--danger)' }}>{err}</div>}
        </div>

        <div style={{ marginTop: 8 }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Paste a long-form task… (or use the demo chips below)"
          />
        </div>

        <div className="row" style={{ marginTop: 10 }}>
          <button className="button" onClick={onCompare} disabled={loading}>
            {loading ? 'Running compare…' : 'Submit & Compare'}
          </button>
          <button
            className="button secondary"
            onClick={() => { setPrompt(''); setOutline([]); setBaseline(''); setPlanText(''); setMB(null); setMP(null); setErr(null); }}
            disabled={loading}
          >
            Clear
          </button>
        </div>

        {/* Demo chips */}
        <div className="row" style={{ marginTop: 12 }}>
          {DEMOS.map((d, i) => (
            <button key={i} className="kbd" onClick={() => pasteDemo(i)} title="Click to paste this example">
              Demo {i + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      <div className="grid two">
        {/* Outline */}
        <div className="card outline">
          <h3 style={{ marginTop: 0 }}>Outline</h3>
          {outline?.length ? (
            <ol>{outline.map((o, i) => <li key={i}>{o}</li>)}</ol>
          ) : (
            <p className="subtle">No outline yet. Submit a prompt to generate one.</p>
          )}
        </div>

        {/* Baseline metrics */}
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Baseline metrics</h3>
          {mb ? (
            <div className="metrics">
              <div className="metric"><div className="label">structure</div><div className="val">{fmt(mb.structure)}</div></div>
              <div className="metric"><div className="label">plan_adherence</div><div className="val">{fmt(mb.plan_adherence)}</div></div>
              <div className="metric"><div className="label">hallucination</div><div className="val">{fmt(mb.hallucination_proxy)}</div></div>
            </div>
          ) : <p className="subtle">—</p>}
        </div>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        {/* Baseline text */}
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Baseline</h3>
          {baseline ? <pre>{baseline}</pre> : <p className="subtle">—</p>}
        </div>

        {/* Plan-first metrics */}
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Plan-first metrics</h3>
          {mp ? (
            <div className="metrics">
              <div className="metric"><div className="label">structure</div><div className="val ok">{fmt(mp.structure)}</div></div>
              <div className="metric"><div className="label">plan_adherence</div><div className="val ok">{fmt(mp.plan_adherence)}</div></div>
              <div className="metric"><div className="label">hallucination</div><div className="val bad">{fmt(mp.hallucination_proxy)}</div></div>
            </div>
          ) : <p className="subtle">—</p>}
        </div>
      </div>

      <div className="grid two" style={{ marginTop: 16 }}>
        {/* Plan-first text */}
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Plan-first</h3>
          {planText ? <pre>{planText}</pre> : <p className="subtle">—</p>}
        </div>

        {/* Notes */}
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Notes</h3>
          <p className="small">
            The <b>hallucination</b> proxy is intentionally strict. You can lower it by asking the model to avoid numbers or
            add lightweight citation tags like <code>[ref]</code> when it uses a number.
          </p>
          <p className="small">
            Backend CORS must allow <code>http://localhost:3000</code>. You already set this in <code>main.py</code>.
          </p>
        </div>
      </div>
    </div>
  );
}
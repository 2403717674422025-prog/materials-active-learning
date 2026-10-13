import { useState, useCallback } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Database, FlaskConical, Brain, Zap, Check,
  ChevronRight, AlertTriangle, RefreshCw, Info,
  Shuffle, Activity, TrendingUp, Sparkles,
} from 'lucide-react'
import { api } from '../services/api'
import type { ExperimentCreate, Strategy, ModelType } from '../types'
import clsx from 'clsx'

const STEPS = [
  { id: 1, label: 'Dataset'  },
  { id: 2, label: 'Budget'   },
  { id: 3, label: 'Model'    },
  { id: 4, label: 'Strategy' },
  { id: 5, label: 'Review'   },
]

const STRATEGIES = [
  { v: 'random'      as Strategy, label: 'Random Sampling',      icon: Shuffle,    desc: 'Uniform random selection — evaluation baseline. No model information used.',                     color: '#5C6E82' },
  { v: 'uncertainty' as Strategy, label: 'Uncertainty Sampling', icon: Activity,   desc: 'Selects highest GP posterior σ(x). Pure exploration — ignores predicted performance.',          color: '#59D5CC' },
  { v: 'ei'          as Strategy, label: 'Expected Improvement',  icon: TrendingUp, desc: 'EI(x) = (μ(x)−y*−ξ)·Φ(Z) + σ(x)·φ(Z). Principled balance of exploitation and exploration.',  color: '#A99BFF' },
  { v: 'puda'        as Strategy, label: 'PUDA (Proposed)',       icon: Sparkles,   desc: 'α·Prediction + β·Uncertainty + γ·Diversity. Greedy k-center diversity prevents batch redundancy.', color: '#41D98C' },
]

const MODELS = [
  { v: 'gp'       as ModelType, label: 'Gaussian Process',  unc: 'True posterior σ(x)',     note: 'Recommended · auto-switches to RF above 800 labeled samples'  },
  { v: 'rf'       as ModelType, label: 'Random Forest',     unc: 'Tree variance proxy',     note: 'Faster · scales to larger labeled sets'                       },
  { v: 'ensemble' as ModelType, label: 'Deep Ensemble',     unc: 'Member std dev',           note: 'Best for N > 1500 labeled samples'                           },
]

const DEF: ExperimentCreate = {
  dataset_id: '', name: '', strategy: 'puda', model_type: 'gp',
  initial_samples: 100, batch_size: 20, total_budget: 300, random_seed: 42,
  target_column: 'gap expt', xi: 0.01, kappa: 2.0,
  alpha: 0.40, beta: 0.35, gamma: 0.25, use_pca: false, pca_n_components: 50,
}

export default function ExperimentSetupPage() {
  const nav = useNavigate()
  const [step, setStep] = useState(1)
  const [form, setForm] = useState<ExperimentCreate>(DEF)
  const [err, setErr] = useState('')
  const [enableTuning, setEnableTuning] = useState(false)
  const [tuningTrials, setTuningTrials] = useState(20)
  const [isLaunching, setIsLaunching] = useState(false)
  const [tuningProgress, setTuningProgress] = useState('')
  const [tuningCounts, setTuningCounts] = useState<{ done: number; total: number } | null>(null)
  const [pendingExperimentId, setPendingExperimentId] = useState<string | null>(null)
  const set = useCallback(<K extends keyof ExperimentCreate>(k: K, v: ExperimentCreate[K]) =>
    setForm(f => ({ ...f, [k]: v })), [])

  const { data: datasets } = useQuery({ queryKey: ['datasets'], queryFn: api.datasets.list })
  const ready = datasets?.filter(d => d.status === 'ready') ?? []
  const ds = ready.find(d => d.id === form.dataset_id)

  const create = useMutation({ mutationFn: (p: ExperimentCreate) => api.experiments.create(p) })
  const init   = useMutation({
    mutationFn: (id: string) => api.experiments.initialize(id),
    onSuccess: (_, id) => nav(`/experiment/${id}`),
    onError: (e: any) => setErr(e?.response?.data?.detail ?? 'Initialization failed'),
  })

  async function launch() {
    setErr('')
    setIsLaunching(true)
    try {
      const ex = await create.mutateAsync(form)
      await init.mutateAsync(ex.id)
      if (enableTuning) {
        if (form.model_type !== 'gp' && form.model_type !== 'rf') {
          throw new Error('Tuning currently supports Gaussian Process and Random Forest only.')
        }
        setPendingExperimentId(ex.id)
        setTuningProgress('Starting Optuna tuning on the initial labeled samples…')
        let run = await api.experiments.startTuning(ex.id, {
          model_type: form.model_type,
          n_trials: tuningTrials,
          random_seed: form.random_seed,
          timeout_seconds: 900,
        })
        setTuningCounts({ done: run.completed_trials, total: run.trial_count })
        while (run.status === 'pending' || run.status === 'running') {
          await new Promise(resolve => window.setTimeout(resolve, 1500))
          run = await api.experiments.tuningStatus(ex.id, run.id)
          setTuningCounts({ done: run.completed_trials, total: run.trial_count })
          setTuningProgress(
            `Validation RMSE tuning: ${run.completed_trials}/${run.trial_count} trials` +
            (run.best_validation_rmse == null ? '' : ` · best ${run.best_validation_rmse.toFixed(4)}`),
          )
        }
        if (run.status !== 'completed') {
          throw new Error(run.error_summary ?? 'Hyperparameter optimization failed.')
        }
        setTuningProgress('Applying the best validation configuration…')
        await api.experiments.applyTuning(ex.id, run.id)
      }
      nav(`/experiment/${ex.id}`)
    } catch (e: any) {
      setErr(e?.response?.data?.detail ?? e?.message ?? 'Failed to initialize experiment')
    } finally {
      setIsLaunching(false)
    }
  }

  const iters = Math.max(1, Math.floor((form.total_budget - form.initial_samples) / Math.max(1, form.batch_size)))
  const poolPct = ds?.num_samples ? ((form.total_budget / ds.num_samples) * 100).toFixed(1) + '%' : '—'
  const canNext: Record<number, boolean> = { 1: !!form.dataset_id, 2: form.total_budget > form.initial_samples, 3: !!form.model_type, 4: !!form.strategy, 5: true }

  return (
    <div className="flex h-full">
      {/* ── Left: steps + form ─────────────────────────── */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Step strip */}
        <div className="flex items-center px-5 py-3 border-b border-line2 bg-ink1 gap-1">
          {STEPS.map((s, i) => {
            const done = step > s.id; const active = step === s.id
            return (
              <div key={s.id} className="flex items-center gap-1">
                <button onClick={() => done && setStep(s.id)}
                  className={clsx('flex items-center gap-1.5 text-[12px] transition-colors', done && 'cursor-pointer')}
                  disabled={!done && !active}>
                  <div className={clsx('w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0',
                    active ? 'bg-teal text-ink' : done ? 'bg-ok text-ink' : 'bg-panel3 text-t3')}>
                    {done ? <Check className="w-3 h-3" /> : s.id}
                  </div>
                  <span className={clsx(active ? 'text-teal font-medium' : done ? 'text-ok' : 'text-t3')}>{s.label}</span>
                </button>
                {i < STEPS.length - 1 && <div className={clsx('w-8 h-px mx-1', done ? 'bg-ok/30' : 'bg-line2')} />}
              </div>
            )
          })}
        </div>

        {/* Form area */}
        <div className="flex-1 overflow-auto px-6 py-6">
          {/* Step 1: Dataset */}
          {step === 1 && (
            <div className="max-w-xl space-y-4">
              <StepHead n={1} title="Dataset & Target" />
              {ready.length === 0 ? (
                <div className="flex items-center gap-3 p-3 card border-amber/20 bg-amber/[0.04] text-[12px]">
                  <AlertTriangle className="w-4 h-4 text-amber flex-shrink-0" />
                  <div><div className="text-t1 font-medium">No datasets available</div>
                    <Link to="/dataset" className="text-teal text-[11px]">Load the expt_gap dataset first →</Link></div>
                </div>
              ) : (
                <div className="space-y-2">
                  {ready.map(d => (
                    <button key={d.id} onClick={() => { set('dataset_id', d.id); set('target_column', d.target_column ?? 'gap expt') }}
                      className={clsx(
                        'w-full text-left p-3.5 rounded card transition-all flex items-center gap-3',
                        form.dataset_id === d.id ? 'border-teal/40 bg-[rgba(89,213,204,0.06)]' : 'hover:border-line',
                      )}>
                      <div className={clsx('w-7 h-7 rounded flex items-center justify-center flex-shrink-0',
                        form.dataset_id === d.id ? 'bg-teal/15' : 'bg-panel2')}>
                        <Database className={clsx('w-3.5 h-3.5', form.dataset_id === d.id ? 'text-teal' : 'text-t3')} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-[13px] font-medium text-t1">{d.name}</div>
                        <div className="text-[11px] text-t3 mt-0.5">{d.num_samples?.toLocaleString()} samples · {d.num_features} features · <span className="text-amber">{d.target_column}</span></div>
                      </div>
                      {form.dataset_id === d.id && <Check className="w-4 h-4 text-teal flex-shrink-0" />}
                    </button>
                  ))}
                </div>
              )}
              <div>
                <div className="t-label block mb-1.5">Experiment name (optional)</div>
                <input className="input text-[12px]" placeholder="e.g. PUDA exploration run 1"
                  value={form.name ?? ''} onChange={e => set('name', e.target.value)} />
              </div>
            </div>
          )}

          {/* Step 2: Budget */}
          {step === 2 && (
            <div className="max-w-xl space-y-5">
              <StepHead n={2} title="Experimental Budget" />
              <div className="grid grid-cols-2 gap-4">
                {[
                  { k: 'initial_samples', l: 'Initial Seed Samples',    hint: 'Labeled before AL begins',    min: 10,  max: 1000 },
                  { k: 'batch_size',      l: 'Batch Size / Iteration',   hint: 'Candidates per AL iteration', min: 1,   max: 200  },
                  { k: 'total_budget',    l: 'Total Budget',             hint: 'Maximum labeled candidates',  min: 20,  max: 5000 },
                  { k: 'random_seed',     l: 'Random Seed',              hint: 'For reproducibility',         min: 0,   max: 99999},
                ].map(({ k, l, hint, min, max }) => (
                  <div key={k}>
                    <div className="t-label block mb-1.5">{l}</div>
                    <input type="number" className="input t-mono text-[13px]" min={min} max={max}
                      value={form[k as keyof ExperimentCreate] as number}
                      onChange={e => set(k as keyof ExperimentCreate, parseInt(e.target.value) as never)} />
                    <div className="text-t4 text-[11px] mt-0.5">{hint}</div>
                  </div>
                ))}
              </div>
              <div className="inset p-3.5 grid grid-cols-4 gap-3 text-center">
                {[
                  { l: 'Iterations', v: iters, c: '#59D5CC'  },
                  { l: 'Pool coverage', v: poolPct, c: '#A99BFF' },
                  { l: 'Seed fraction', v: `${((form.initial_samples/form.total_budget)*100).toFixed(0)}%`, c: '#EDAB60' },
                  { l: 'Acquisitions', v: form.total_budget - form.initial_samples, c: '#41D98C' },
                ].map(({ l, v, c }) => (
                  <div key={l}><div className="t-mono font-bold" style={{ color: c }}>{v}</div><div className="t-label mt-0.5">{l}</div></div>
                ))}
              </div>
              {form.total_budget <= form.initial_samples && (
                <div className="text-err text-[12px] flex items-center gap-1.5"><AlertTriangle className="w-3.5 h-3.5" />Budget must exceed initial samples</div>
              )}
            </div>
          )}

          {/* Step 3: Model */}
          {step === 3 && (
            <div className="max-w-xl space-y-3">
              <StepHead n={3} title="Surrogate Model" />
              {MODELS.map(m => (
                <button key={m.v} onClick={() => {
                  set('model_type', m.v)
                  if (m.v !== 'gp' && m.v !== 'rf') setEnableTuning(false)
                }}
                  className={clsx('w-full text-left p-4 rounded card transition-all flex items-start gap-3',
                    form.model_type === m.v ? 'border-teal/40 bg-[rgba(89,213,204,0.05)]' : 'hover:border-line')}>
                  <div className={clsx('w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 mt-0.5 transition-all',
                    form.model_type === m.v ? 'border-teal bg-teal' : 'border-line')}>
                    {form.model_type === m.v && <Check className="w-3 h-3 text-ink" />}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] font-medium text-t1">{m.label}</span>
                      <span className="chip chip-teal text-[10px]">{m.unc}</span>
                    </div>
                    <div className="text-[11px] text-t3 mt-0.5">{m.note}</div>
                  </div>
                </button>
              ))}
              <div className="inset p-3 flex items-start gap-2 text-[11px] text-t3">
                <Info className="w-3.5 h-3.5 text-teal flex-shrink-0 mt-0.5" />
                GP provides true Bayesian posterior σ(x), not a heuristic. The uncertainty is the actual mathematical posterior standard deviation used in EI/PUDA.
              </div>
              <div className="card rounded p-4 space-y-3">
                <label className="flex items-start gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    className="mt-0.5 accent-teal"
                    checked={enableTuning}
                    disabled={form.model_type !== 'gp' && form.model_type !== 'rf'}
                    onChange={e => setEnableTuning(e.target.checked)}
                  />
                  <span>
                    <span className="block text-[13px] font-medium text-t1">Optimize model hyperparameters with Optuna</span>
                    <span className="block text-[11px] text-t3 mt-0.5">
                      Uses only the initial labeled training-pool samples and an internal validation split. Final test data is not used to select parameters.
                    </span>
                  </span>
                </label>
                {enableTuning && (
                  <div className="grid grid-cols-2 gap-3 border-t border-line2 pt-3">
                    <label className="text-[11px] text-t3">
                      Trial budget
                      <input
                        type="number"
                        className="input t-mono text-[12px] mt-1"
                        min={1}
                        max={100}
                        value={tuningTrials}
                        onChange={e => setTuningTrials(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
                      />
                    </label>
                    <div className="text-[11px] text-t3">
                      Random seed
                      <div className="input t-mono text-[12px] mt-1">{form.random_seed}</div>
                    </div>
                    <div className="col-span-2 text-[10px] text-t4">
                      Search method: Optuna TPE · objective: validation RMSE · time limit: 15 minutes
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 4: Strategy */}
          {step === 4 && (
            <div className="max-w-xl space-y-3">
              <StepHead n={4} title="Acquisition Strategy" />
              {STRATEGIES.map(s => {
                const Icon = s.icon; const active = form.strategy === s.v
                return (
                  <div key={s.v} className={clsx('card rounded overflow-hidden transition-all',
                    active ? 'border-l-2' : 'border-l-2 border-l-transparent')}
                    style={active ? { borderLeftColor: s.color } : {}}>
                    <button onClick={() => set('strategy', s.v)}
                      className="w-full text-left p-4 flex items-start gap-3">
                      <div className={clsx('w-7 h-7 rounded flex items-center justify-center flex-shrink-0 mt-0.5 transition-all',
                        active ? 'bg-white/[0.08]' : 'bg-panel2')}>
                        <Icon style={{ width: 14, height: 14, color: active ? s.color : '#5C6E82' }} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-[13px] font-medium text-t1">{s.label}</span>
                        </div>
                        <div className="text-[11px] text-t3 mt-0.5 leading-relaxed">{s.desc}</div>
                      </div>
                      <div className={clsx('w-4 h-4 rounded-full border flex items-center justify-center flex-shrink-0 mt-0.5',
                        active ? 'border-teal bg-teal' : 'border-line')}>
                        {active && <Check className="w-2.5 h-2.5 text-ink" />}
                      </div>
                    </button>
                    {active && s.v === 'puda' && (
                      <div className="px-4 pb-4 pt-0 border-t border-line2">
                        <div className="t-label mb-2 mt-2">Weights — α + β + γ = 1.0</div>
                        <div className="grid grid-cols-3 gap-3">
                          {[
                            { k: 'alpha' as const, l: 'α Performance', c: '#59D5CC' },
                            { k: 'beta'  as const, l: 'β Uncertainty',  c: '#A99BFF' },
                            { k: 'gamma' as const, l: 'γ Diversity',   c: '#EDAB60' },
                          ].map(({ k, l, c }) => (
                            <div key={k}>
                              <div className="t-label block mb-1" style={{ color: c }}>{l}</div>
                              <input type="number" className="input t-mono text-[12px]" step="0.05" min="0" max="1"
                                value={form[k]} onChange={e => set(k, parseFloat(e.target.value) || 0)} />
                            </div>
                          ))}
                        </div>
                        <WeightBar a={form.alpha} b={form.beta} g={form.gamma} />
                      </div>
                    )}
                    {active && s.v === 'ei' && (
                      <div className="px-4 pb-4 pt-0 border-t border-line2">
                        <div className="t-label block mb-1.5 mt-2">ξ — Exploration jitter</div>
                        <input type="number" className="input w-28 t-mono text-[12px]" step="0.001" min="0" max="1"
                          value={form.xi} onChange={e => set('xi', parseFloat(e.target.value) || 0)} />
                        <div className="text-t4 text-[11px] mt-1">Higher = more exploration. Typical: 0.001–0.1</div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          {/* Step 5: Review */}
          {step === 5 && (
            <div className="max-w-xl space-y-4">
              <StepHead n={5} title="Review & Launch" />
              <div className="card divide-y divide-line2">
                {[
                  { l: 'Dataset',   v: ds?.name ?? form.dataset_id.slice(0,16) + '…' },
                  { l: 'Target',    v: form.target_column },
                  { l: 'Model',     v: MODELS.find(m => m.v === form.model_type)?.label ?? form.model_type },
                  { l: 'Strategy',  v: STRATEGIES.find(s => s.v === form.strategy)?.label ?? form.strategy },
                  { l: 'Initial',   v: `${form.initial_samples} labeled samples` },
                  { l: 'Batch',     v: `${form.batch_size} per iteration` },
                  { l: 'Budget',    v: `${form.total_budget} total (${iters} iterations)` },
                  { l: 'Seed',      v: String(form.random_seed) },
                  ...(enableTuning ? [{ l: 'Hyperparameter tuning', v: `Optuna · ${tuningTrials} trials` }] : []),
                  ...(form.strategy === 'puda' ? [
                    { l: 'α (perf)', v: String(form.alpha) },
                    { l: 'β (unc)',  v: String(form.beta)  },
                    { l: 'γ (div)',  v: String(form.gamma) },
                  ] : []),
                  ...(form.strategy === 'ei' ? [{ l: 'ξ', v: String(form.xi) }] : []),
                ].map(({ l, v }) => (
                  <div key={l} className="flex items-center justify-between px-4 py-2.5 text-[12px]">
                    <span className="text-t3">{l}</span>
                    <span className="t-mono text-t1 font-medium">{v}</span>
                  </div>
                ))}
              </div>
              {err && (
                <div className="flex items-center gap-2 p-3 card border-err/30 text-err text-[12px]">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />{err}
                </div>
              )}
              {isLaunching && enableTuning && (
                <div className="card p-3 text-[12px] text-t3 space-y-1">
                  <div className="flex items-center gap-2 text-teal">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>{tuningProgress || 'Preparing tuning run…'}</span>
                  </div>
                  {tuningCounts && (
                    <div className="text-[11px]">
                      Best-so-far validation RMSE and trial details are saved as the run progresses.
                    </div>
                  )}
                </div>
              )}
              {pendingExperimentId && err && !isLaunching && (
                <button className="btn btn-ghost w-full justify-center text-[12px]"
                  onClick={() => nav(`/experiment/${pendingExperimentId}`)}>
                  Continue with manual model settings
                </button>
              )}
              <button className="btn btn-teal w-full py-2.5 text-[13px] font-semibold justify-center"
                onClick={launch} disabled={create.isPending || init.isPending || isLaunching}>
                {create.isPending || init.isPending || isLaunching
                  ? <><RefreshCw className="w-4 h-4 animate-spin" />{enableTuning ? 'Tuning model…' : 'Initializing…'}</>
                  : <><Zap className="w-4 h-4" />{enableTuning ? 'Start Optimization & Experiment' : 'Initialize Experiment'}</>}
              </button>
            </div>
          )}
        </div>

        {/* Nav footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-line2 bg-ink1">
          <button className="btn btn-ghost text-[12px]" onClick={() => setStep(s => Math.max(1, s-1))} disabled={step === 1}>← Back</button>
          {step < 5 && (
            <button className="btn btn-teal text-[12px]" onClick={() => setStep(s => Math.min(5, s+1))} disabled={!canNext[step]}>
              Continue <ChevronRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* ── Right: live summary rail ──────────────────────── */}
      <div className="w-[220px] flex-shrink-0 border-l border-line2 bg-ink1 flex flex-col">
        <div className="px-4 pt-4 pb-3 border-b border-line2"><div className="t-section">Configuration</div></div>
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 text-[11px]">
          {[
            { l: 'Dataset',   v: ds?.name ?? '—',            c: !!ds },
            { l: 'Samples',   v: ds?.num_samples?.toLocaleString() ?? '—', c: !!ds },
            { l: 'Target',    v: form.target_column || '—',  c: !!form.target_column },
            { l: 'Model',     v: MODELS.find(m => m.v === form.model_type)?.label ?? '—', c: true },
            { l: 'Strategy',  v: STRATEGIES.find(s => s.v === form.strategy)?.label ?? '—', c: true },
            { l: 'Budget',    v: `${form.total_budget}`,     c: true },
            { l: 'Batch',     v: `${form.batch_size} / iter`,c: true },
            { l: '≈ Iters',   v: `${iters}`,                 c: true },
            { l: 'Pool cov.', v: poolPct,                    c: !!ds  },
            { l: 'Seed',      v: String(form.random_seed),   c: true },
          ].map(({ l, v, c }) => (
            <div key={l} className="flex items-start justify-between gap-2">
              <span className="text-t4 flex-shrink-0">{l}</span>
              <span className={clsx('t-mono text-right', c ? 'text-t1' : 'text-t4')}>{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StepHead({ n, title }: { n: number; title: string }) {
  return (
    <div className="border-b border-line2 pb-3 mb-4">
      <div className="t-label text-teal mb-0.5">Step {n} of 5</div>
      <div className="t-title text-[14px]">{title}</div>
    </div>
  )
}

function WeightBar({ a, b, g }: { a: number; b: number; g: number }) {
  const total = a + b + g || 1
  const sum = +(a + b + g).toFixed(2)
  const ok = sum >= 0.99 && sum <= 1.01
  return (
    <div className="mt-3 space-y-1.5">
      <div className="h-1.5 flex rounded-sm overflow-hidden gap-px">
        {[{ v: a/total, c: '#59D5CC' },{ v: b/total, c: '#A99BFF' },{ v: g/total, c: '#EDAB60' }].map((bar, i) => (
          <div key={i} style={{ width:`${bar.v*100}%`, background: bar.c, opacity: 0.8 }} className="h-full" />
        ))}
      </div>
      <div className="flex justify-between">
        <div className="flex gap-3 text-[10px] text-t3">
          {[{l:'α',c:'#59D5CC',v:a},{l:'β',c:'#A99BFF',v:b},{l:'γ',c:'#EDAB60',v:g}].map(({l,c,v})=>(
            <span key={l}><span style={{color:c}}>{l}</span>={v.toFixed(2)}</span>
          ))}
        </div>
        <span className={clsx('t-mono text-[11px] font-semibold', ok ? 'text-ok' : 'text-err')}>Σ={sum}</span>
      </div>
    </div>
  )
}

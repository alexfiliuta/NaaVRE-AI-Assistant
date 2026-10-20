import { ReactWidget } from '@jupyterlab/apputils';
import React, { useEffect, useRef, useState } from 'react';

import type {
  AgentContext, AgentRun, AgentStatus, AssistantAnalysis, AssistantClient,
  HealthResponse, KnowledgeItem, Lab, LabKnowledge, SupportedLanguage
} from '../api/client';

export interface AssistantSubmission {
  prompt: string;
  language: SupportedLanguage;
  executeAndHeal: boolean;
  includeNotebookContext: boolean;
  generateVisualization: boolean;
  labId?: string;
  agentContext?: AgentContext;
}

interface AssistantPanelOptions {
  client: AssistantClient;
  checkHealth: () => Promise<HealthResponse>;
  getEnvironmentContext: (includeNotebook: boolean, language: SupportedLanguage) => Record<string, unknown>;
  onSubmit: (submission: AssistantSubmission) => Promise<void>;
}

export class AssistantPanel extends ReactWidget {
  private externalStatus: { message: string; isError: boolean } | null = null;
  private externalAnalysis: AssistantAnalysis | null = null;

  constructor(private readonly options: AssistantPanelOptions) {
    super();
    this.addClass('naavre-AIAssistant');
  }

  setStatus(message: string, isError = false): void {
    this.externalStatus = { message, isError };
    this.update();
  }

  setAnalysis(analysis: AssistantAnalysis | null): void {
    this.externalAnalysis = analysis;
    this.update();
  }

  protected render(): React.ReactElement {
    return <AssistantView {...this.options} externalStatus={this.externalStatus} externalAnalysis={this.externalAnalysis} />;
  }
}

interface AssistantViewProps extends AssistantPanelOptions {
  externalStatus: { message: string; isError: boolean } | null;
  externalAnalysis: AssistantAnalysis | null;
}

const MAX_DOCUMENT_BYTES = 4 * 1024 * 1024;
const STATUS_LABELS: Record<AgentStatus, string> = {
  queued: 'Queued', running: 'Investigating', awaiting_approval: 'Approval needed',
  completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', exhausted: 'Investigation safety limit reached'
};

function isActive(status: AgentStatus): boolean {
  return status === 'queued' || status === 'running' || status === 'awaiting_approval';
}

function AssistantView({
  client, checkHealth, getEnvironmentContext, onSubmit, externalStatus, externalAnalysis
}: AssistantViewProps): React.ReactElement {
  const [mode, setMode] = useState<'research' | 'notebook'>('research');
  const [prompt, setPrompt] = useState('');
  const [language, setLanguage] = useState<SupportedLanguage>('python');
  const [executeAndHeal, setExecuteAndHeal] = useState(false);
  const [includeNotebookContext, setIncludeNotebookContext] = useState(true);
  const [generateVisualization, setGenerateVisualization] = useState(true);
  const [isWorking, setIsWorking] = useState(false);
  const [status, setStatus] = useState('Ready to build reviewable notebook cells.');
  const [isError, setIsError] = useState(false);
  const [health, setHealth] = useState('Checking server configuration…');
  const [healthError, setHealthError] = useState(false);
  const [analysis, setAnalysis] = useState<AssistantAnalysis | null>(null);
  const [labs, setLabs] = useState<Lab[]>([]);
  const [labId, setLabId] = useState('');
  const [labsLoading, setLabsLoading] = useState(true);
  const [labsError, setLabsError] = useState('');
  const [labsVersion, setLabsVersion] = useState(0);
  const [knowledge, setKnowledge] = useState<LabKnowledge | null>(null);
  const [knowledgeQuery, setKnowledgeQuery] = useState('');
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [knowledgeError, setKnowledgeError] = useState('');
  const [knowledgeVersion, setKnowledgeVersion] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [uploadError, setUploadError] = useState(false);
  const [run, setRun] = useState<AgentRun | null>(null);
  const [starting, setStarting] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [agentError, setAgentError] = useState('');
  const [runLoaded, setRunLoaded] = useState(false);
  const [pollVersion, setPollVersion] = useState(0);
  const [revision, setRevision] = useState(0);
  const [runRevision, setRunRevision] = useState(-1);
  const [runPrompt, setRunPrompt] = useState('');
  const mounted = useRef(true);
  const revisionRef = useRef(0);
  const startLock = useRef(false);
  const actionLock = useRef(false);
  const buildLock = useRef(false);
  const uploadLock = useRef(false);
  const currentRunId = useRef('');
  // Invalidates reads already in flight before a decision/cancel is submitted.
  const mutationVersion = useRef(0);

  const lab = labs.find(item => item.id === labId);
  const activeRun = !!run && isActive(run.status);
  const staleRun = !!run && (runRevision !== revision || run.lab_id !== labId);
  const usableRun = !!run && !!lab && runLoaded && !staleRun && run.status === 'completed';
  const requestLocked = isWorking || starting;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    let disposed = false;
    void checkHealth().then(result => {
      if (disposed) { return; }
      setHealthError(!result.configured);
      const roleModels = result.models;
      const modelDescription = roleModels && new Set(Object.values(roleModels)).size > 1
        ? 'Configured models: investigation ' + roleModels.agent + ', planning ' + roleModels.analysis + ', coding ' + roleModels.code
        : 'Configured model: ' + result.model;
      setHealth(result.configured
        ? modelDescription + (result.schemaContract ? ' · ' + result.schemaContract : '') + '. Live inference has not been checked.'
        : result.configurationError ?? 'Server configuration is incomplete.');
    }).catch(error => {
      if (!disposed) {
        setHealthError(true);
        setHealth('Configuration check unavailable: ' + errorMessage(error));
      }
    });
    return () => { disposed = true; };
  }, [checkHealth]);

  useEffect(() => {
    const controller = new AbortController();
    setLabsLoading(true);
    setLabsError('');
    void client.labs(controller.signal).then(items => {
      if (controller.signal.aborted) { return; }
      setLabs(items);
      setLabId(previous => previous || items[0]?.id || '');
    }).catch(error => {
      if (!controller.signal.aborted) {
        setLabsError('Lab workspace unavailable: ' + errorMessage(error) + '. Notebook generation is still available.');
      }
    }).finally(() => {
      if (!controller.signal.aborted) { setLabsLoading(false); }
    });
    return () => controller.abort();
  }, [client, labsVersion]);

  useEffect(() => {
    const controller = new AbortController();
    setKnowledge(null);
    setKnowledgeError('');
    setKnowledgeLoading(!!labId);
    if (!labId) { return () => controller.abort(); }
    const timer = window.setTimeout(() => {
      void client.knowledge(labId, knowledgeQuery.trim(), controller.signal).then(result => {
        if (!controller.signal.aborted) { setKnowledge(result); }
      }).catch(error => {
        if (!controller.signal.aborted) { setKnowledgeError(errorMessage(error)); }
      }).finally(() => {
        if (!controller.signal.aborted) { setKnowledgeLoading(false); }
      });
    }, 300);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [client, labId, knowledgeQuery, knowledgeVersion]);

  const runId = run?.run_id;
  useEffect(() => {
    // POST may return a terminal status but does not include the evidence.
    if (!runId || actionBusy || (runLoaded && !activeRun)) { return; }
    const controller = new AbortController();
    const version = mutationVersion.current;
    let timer: number | undefined;
    let failures = 0;
    const poll = async (): Promise<void> => {
      try {
        const next = await client.agentRun(runId, controller.signal);
        if (controller.signal.aborted || version !== mutationVersion.current || currentRunId.current !== runId) { return; }
        if (next.run_id !== runId) { throw new Error('Server returned a different investigation.'); }
        setRun(next);
        setRunLoaded(true);
        setAgentError('');
        failures = 0;
        if (!isActive(next.status)) { return; }
      } catch (error) {
        if (controller.signal.aborted || version !== mutationVersion.current) { return; }
        failures += 1;
        setAgentError('Could not refresh investigation: ' + errorMessage(error) + '. Retrying; the server run may still be active.');
      }
      timer = window.setTimeout(() => { void poll(); }, Math.min(1500 * 2 ** failures, 15000));
    };
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [client, runId, activeRun, runLoaded, actionBusy, pollVersion]);

  useEffect(() => {
    if (externalStatus) {
      setStatus(externalStatus.message);
      setIsError(externalStatus.isError);
    }
  }, [externalStatus]);

  useEffect(() => { setAnalysis(externalAnalysis); }, [externalAnalysis]);

  const invalidateContext = (): void => {
    revisionRef.current += 1;
    setRevision(revisionRef.current);
    setAnalysis(null);
  };

  const changeLab = (id: string): void => {
    invalidateContext();
    setLabId(id);
    setKnowledge(null);
    setKnowledgeQuery('');
    setUploadStatus('');
  };

  const upload = async (file: File): Promise<void> => {
    if (!labId || uploadLock.current) { return; }
    setUploadError(false);
    if (!/\.(pdf|txt|md)$/i.test(file.name) || file.size === 0 || file.size > MAX_DOCUMENT_BYTES) {
      setUploadError(true);
      setUploadStatus('Choose a nonempty PDF, TXT, or MD knowledge document, at most 4 MiB.');
      return;
    }
    uploadLock.current = true;
    setUploading(true);
    setUploadStatus('Uploading ' + file.name + '…');
    try {
      const content = await documentBase64(file);
      await client.uploadDocument(labId, file.name, content);
      if (!mounted.current) { return; }
      invalidateContext();
      setUploadStatus(file.name + ' added to lab knowledge.');
      setKnowledgeVersion(value => value + 1);
      setLabsVersion(value => value + 1);
    } catch (error) {
      if (mounted.current) { setUploadError(true); setUploadStatus(errorMessage(error)); }
    } finally {
      uploadLock.current = false;
      if (mounted.current) { setUploading(false); }
    }
  };

  const investigate = async (): Promise<void> => {
    if (!lab || !prompt.trim() || activeRun || startLock.current || uploadLock.current || buildLock.current) { return; }
    if (prompt.trim().length > 8000) {
      setAgentError('Lab investigations accept requests up to 8,000 characters. Shorten the request or build notebook cells directly.');
      return;
    }
    startLock.current = true;
    setStarting(true);
    setAgentError('');
    const requestRevision = revisionRef.current;
    try {
      const created = await client.startAgentRun({
        lab_id: labId, description: prompt.trim(),
        context: getEnvironmentContext(includeNotebookContext, language)
      });
      if (!mounted.current) { return; }
      currentRunId.current = created.run_id;
      setRun({ ...created, lab_id: labId, events: [] });
      setRunLoaded(false);
      setRunRevision(requestRevision);
      setRunPrompt(prompt.trim());
      setPollVersion(value => value + 1);
    } catch (error) {
      if (mounted.current) { setAgentError(errorMessage(error)); }
    } finally {
      startLock.current = false;
      if (mounted.current) { setStarting(false); }
    }
  };

  const actOnRun = async (action: 'approve' | 'reject' | 'cancel'): Promise<void> => {
    if (!run || !activeRun || actionLock.current || (action === 'approve' && staleRun)) { return; }
    actionLock.current = true;
    mutationVersion.current += 1;
    setActionBusy(true);
    setAgentError('');
    try {
      if (action === 'cancel') { await client.cancelAgentRun(run.run_id); }
      else { await client.decideAgentRun(run.run_id, action === 'approve'); }
    } catch (error) {
      if (mounted.current) { setAgentError(errorMessage(error) + ' The run will be refreshed before another action.'); }
    } finally {
      actionLock.current = false;
      if (mounted.current) {
        setActionBusy(false);
        setRunLoaded(false);
        setPollVersion(value => value + 1);
      }
    }
  };

  const submit = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault();
    if (!prompt.trim() || buildLock.current || startLock.current || uploadLock.current) { return; }
    buildLock.current = true;
    setIsWorking(true);
    setIsError(false);
    setAnalysis(null);
    setStatus('Building notebook cells…');
    try {
      await onSubmit({
        prompt: prompt.trim(), language, executeAndHeal, includeNotebookContext, generateVisualization,
        labId: labId || undefined,
        agentContext: usableRun && run ? {
          ...run.context,
          agent_run_id: run.run_id
        } : undefined
      });
      if (mounted.current) {
        setStatus(executeAndHeal
          ? 'Cell generation and opted-in execution finished. Review the code and outputs.'
          : 'Cells inserted in the active notebook. Review them, then run when ready.');
      }
    } catch (error) {
      if (mounted.current) { setIsError(true); setStatus(errorMessage(error)); }
    } finally {
      buildLock.current = false;
      if (mounted.current) { setIsWorking(false); }
    }
  };

  return (
    <div className="naavre-AIAssistant-content">
      <header className="naavre-AIAssistant-header">
        <span className="naavre-AIAssistant-eyebrow">NaaVRE · Research workspace</span>
        <h2>Research coworker</h2>
        <p>Investigate with your lab. Build cells you can review.</p>
      </header>

      <div className="naavre-AIAssistant-modes" role="group" aria-label="Assistant mode">
        <button type="button" aria-pressed={mode === 'research'} onClick={() => setMode('research')}>Lab research</button>
        <button type="button" aria-pressed={mode === 'notebook'} onClick={() => setMode('notebook')}>Notebook builder</button>
      </div>

      {mode === 'research' && (
        <section className="naavre-AIAssistant-card" aria-labelledby="naavre-workspace-title">
          <div className="naavre-AIAssistant-section-heading">
            <h3 id="naavre-workspace-title">Lab workspace</h3>
            <button type="button" className="naavre-AIAssistant-text-button" disabled={labsLoading || requestLocked || uploading} onClick={() => setLabsVersion(value => value + 1)}>Refresh labs</button>
          </div>
          <label htmlFor="naavre-lab">Selected lab</label>
          <select id="naavre-lab" value={labId} disabled={labsLoading || requestLocked || uploading} onChange={event => changeLab(event.target.value)}>
            <option value="">{labsLoading ? 'Loading labs…' : 'No lab selected'}</option>
            {labs.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          {labsError && <p className="is-error" role="alert">{labsError}</p>}
          {!labsLoading && !labsError && !labs.length && <p>No labs are available. You can still build notebook cells below.</p>}
          {lab && <>
            <p>{lab.description || 'No lab description provided.'}</p>
            <dl className="naavre-AIAssistant-lab-details">
              <dt>Frameworks</dt><dd>{lab.frameworks?.join(', ') || 'Not specified'}</dd>
              <dt>Working style</dt><dd>{lab.style || 'Not specified'}</dd>
            </dl>
            <details className="naavre-AIAssistant-knowledge" open>
              <summary>Lab knowledge · {lab.document_count} documents · {lab.skill_count} skills</summary>
              <label htmlFor="naavre-knowledge-search">Find documents and skills</label>
              <input id="naavre-knowledge-search" type="search" maxLength={500} value={knowledgeQuery} onChange={event => setKnowledgeQuery(event.target.value)} placeholder="Search lab knowledge" />
              {knowledgeLoading && <p role="status">Loading knowledge…</p>}
              {knowledgeError && <p className="is-error" role="alert">{knowledgeError} <button type="button" onClick={() => setKnowledgeVersion(value => value + 1)}>Retry</button></p>}
              {knowledge && <>
                <KnowledgeList title={knowledgeQuery.trim() ? 'Document matches' : 'Documents'} items={knowledge.hits ?? knowledge.documents ?? []} />
                <KnowledgeList title="Skills" items={knowledge.skills ?? []} />
                {knowledge.truncated && <small>Showing a bounded selection of lab knowledge. Refine the search for more specific results.</small>}
              </>}
              <label htmlFor="naavre-knowledge-upload">Add lab knowledge</label>
              <input id="naavre-knowledge-upload" type="file" accept=".pdf,.txt,.md" aria-describedby="naavre-upload-help" disabled={uploading || requestLocked || activeRun} onChange={event => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = '';
                if (file) { void upload(file); }
              }} />
              <small id="naavre-upload-help">PDF, TXT or MD · up to 4 MiB each. Upload protocols and reference documents here. This does not upload datasets.</small>
              {uploadStatus && <p className={uploadError ? 'is-error' : ''} role="status">{uploadStatus}</p>}
            </details>
          </>}
        </section>
      )}

      <form className="naavre-AIAssistant-form naavre-AIAssistant-card" onSubmit={submit}>
        <div className="naavre-AIAssistant-section-heading"><h3>Research request</h3><span className="naavre-AIAssistant-badge">{lab?.name ?? 'No lab'}</span></div>
        <label htmlFor="naavre-ai-prompt">What would you like to work on?</label>
        <textarea id="naavre-ai-prompt" value={prompt} rows={6} disabled={requestLocked}
          onChange={event => { invalidateContext(); setPrompt(event.target.value); }}
          placeholder="Explore /data/observations.csv, check the temperature measurements, and plan a reproducible analysis."
          aria-describedby="naavre-source-help" />
        <small id="naavre-source-help">Data paths refer to the Jupyter server, which may be remote. A path on your computer is not a dataset upload.</small>
        <fieldset className="naavre-AIAssistant-language" disabled={requestLocked}>
          <legend>Notebook language</legend>
          {(['python', 'r'] as const).map(value => <label key={value}>
            <input type="radio" name="naavre-language" checked={language === value} onChange={() => { invalidateContext(); setLanguage(value); }} />
            {value === 'python' ? 'Python' : 'R'}
          </label>)}
        </fieldset>
        <label className="naavre-AIAssistant-toggle">
          <input type="checkbox" checked={includeNotebookContext} disabled={requestLocked} onChange={event => { invalidateContext(); setIncludeNotebookContext(event.target.checked); }} />
          Include notebook cells and declared file references
        </label>
        <small>Uses bounded context near the active cell. Existing cells are kept.</small>

        {mode === 'research' && <div className="naavre-AIAssistant-step">
          <h4>1 · Investigate <span className="naavre-AIAssistant-muted">Optional</span></h4>
          <p>Actions execute through lab tools on the server. Review any requested tool approval below.</p>
          <button type="button" className="naavre-AIAssistant-secondary" disabled={!lab || !prompt.trim() || requestLocked || activeRun || uploading || labsLoading} onClick={() => { void investigate(); }}>
            {starting ? 'Starting investigation…' : 'Investigate with lab tools'}
          </button>
          {activeRun && <small>An investigation is active. Wait for it to finish or cancel it before starting another.</small>}
        </div>}

        <div className="naavre-AIAssistant-step">
          <h4>{mode === 'research' ? '2 · ' : ''}Build notebook cells</h4>
          <p>{usableRun ? 'The completed investigation’s summary, evidence and context will be included.' : 'Build directly from your request. No completed, current investigation context will be included.'}</p>
          <label className="naavre-AIAssistant-toggle">
            <input type="checkbox" checked={generateVisualization} disabled={requestLocked} onChange={event => setGenerateVisualization(event.target.checked)} />
            Plan a visualization when appropriate
          </label>
          <label className="naavre-AIAssistant-toggle">
            <input type="checkbox" checked={executeAndHeal} disabled={requestLocked} onChange={event => setExecuteAndHeal(event.target.checked)} />
            Also execute cells and repair errors (up to 3 repairs per cell)
          </label>
          <small>{executeAndHeal ? 'You have opted in: building will also run the generated cells in the notebook kernel.' : 'Generated cells run only when you run them or explicitly opt in above.'}</small>
          <button type="submit" className="naavre-AIAssistant-primary" disabled={requestLocked || uploading || !prompt.trim()}>{isWorking ? 'Building notebook cells…' : 'Build notebook cells'}</button>
        </div>
      </form>

      {agentError && <p className="naavre-AIAssistant-status is-error" role="alert">{agentError}</p>}
      {run && <section className="naavre-AIAssistant-card" aria-labelledby="naavre-investigation-title">
        <div className="naavre-AIAssistant-section-heading">
          <h3 id="naavre-investigation-title">Lab investigation</h3>
          <span className={'naavre-AIAssistant-badge status-' + run.status} role="status">{STATUS_LABELS[run.status] ?? run.status}</span>
        </div>
        <p className="naavre-AIAssistant-muted">{labs.find(item => item.id === run.lab_id)?.name ?? run.lab_id} · Run {run.run_id}</p>
        <p className="naavre-AIAssistant-run-prompt">{runPrompt}</p>
        {staleRun && <p className="naavre-AIAssistant-notice" role="status">Out of date: the lab, request, language, notebook context setting or lab knowledge changed. This run cannot be approved or used to build cells. Start a new investigation for current evidence.</p>}
        {!runLoaded && <p role="status">Refreshing investigation state…</p>}
        {activeRun && <button type="button" disabled={actionBusy} onClick={() => { void actOnRun('cancel'); }}>{actionBusy ? 'Sending action…' : 'Cancel investigation'}</button>}
        {run.status === 'awaiting_approval' && run.pending_action && <section className="naavre-AIAssistant-approval" aria-label="Pending lab tool approval">
          <h4>Approve a lab tool action</h4>
          <p>This approval allows the lab tool <strong>{run.pending_action.tool}</strong> to act. It does not run generated notebook cells.</p>
          <pre>{displayValue(run.pending_action)}</pre>
          <div className="naavre-AIAssistant-actions">
            <button type="button" className="naavre-AIAssistant-primary" disabled={actionBusy || staleRun || !runLoaded || !!agentError} onClick={() => { void actOnRun('approve'); }}>Approve tool action</button>
            <button type="button" disabled={actionBusy || !runLoaded || !!agentError} onClick={() => { void actOnRun('reject'); }}>Reject</button>
          </div>
        </section>}
        <h4>Action timeline</h4>
        {run.events?.length ? <ol className="naavre-AIAssistant-timeline">
          {run.events.map((item, index) => <li key={run.run_id + '-' + index}>
            <div><span className="naavre-AIAssistant-event-kind">{item.kind.replace(/_/g, ' ')}</span>{item.tool && <code>{item.tool}</code>}</div>
            <p>{item.message}</p>
            {item.citation && <small className="naavre-AIAssistant-citation">{item.citation}</small>}
            {item.result !== undefined && <details><summary>Tool result / evidence</summary><pre>{displayValue(item.result)}</pre></details>}
          </li>)}
        </ol> : <p className="naavre-AIAssistant-muted">No actions reported yet.</p>}
        <h4>Summary and evidence</h4>
        <p className="naavre-AIAssistant-run-summary">{run.summary || 'No summary reported yet.'}</p>
        {run.context && Object.keys(run.context).length > 0 && <details><summary>Returned investigation context</summary><pre>{displayValue(run.context)}</pre></details>}
        {run.status === 'exhausted' && <p>The investigation reached its configured step or time limit, not a paid-credit limit. Review the partial results; they will not be used as completed investigation context.</p>}
        {(run.status === 'failed' || run.status === 'cancelled') && <p>This investigation will not be included when building cells.</p>}
      </section>}

      {analysis && <AnalysisSummary analysis={analysis} />}
      <div className={'naavre-AIAssistant-status' + (isError ? ' is-error' : '')} role="status" aria-live="polite">{status}</div>
      <details className="naavre-AIAssistant-health"><summary>Server configuration</summary><p className={healthError ? 'is-error' : ''}>{health}</p></details>
    </div>
  );
}

function KnowledgeList({ title, items }: { title: string; items: Array<KnowledgeItem | string> }): React.ReactElement {
  return <div className="naavre-AIAssistant-knowledge-list">
    <h4>{title}</h4>
    {items.length ? <ul>{items.map((item, index) => {
      const name = typeof item === 'string' ? item : item.filename ?? item.name ?? item.title ?? item.path ?? item.citation?.path ?? item.id ?? title + ' ' + (index + 1);
      const description = typeof item === 'string' ? '' : item.description ?? item.summary;
      return <li key={index}>
        <span>{name}</span>{description && <small>{description}</small>}
        {typeof item !== 'string' && <>
          {item.bytes !== undefined && <small>{item.bytes.toLocaleString()} bytes</small>}
          {item.score !== undefined && <small>Match score: {item.score}</small>}
          {item.text && <p className="naavre-AIAssistant-knowledge-excerpt">{item.text}</p>}
          {(item.content || item.preview) && <details><summary>{item.truncated ? 'Truncated preview' : 'Read skill'}</summary><pre>{item.content ?? item.preview}</pre></details>}
          {item.citation && <small className="naavre-AIAssistant-citation">
            {item.citation.path}{item.citation.page != null ? ' · page ' + item.citation.page : ''}
            {item.citation.hash && <span title={item.citation.hash}> · SHA-256 {item.citation.hash}</span>}
          </small>}
        </>}
      </li>;
    })}</ul> : <p className="naavre-AIAssistant-muted">No matching {title.toLowerCase()}.</p>}
  </div>;
}

function documentBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Could not read this document.'));
    reader.onabort = () => reject(new Error('Document reading was cancelled.'));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string' || !result.includes(',')) { reject(new Error('Could not encode this document.')); }
      else { resolve(result.slice(result.indexOf(',') + 1)); }
    };
    reader.readAsDataURL(file);
  });
}

function displayValue(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? '';
}

function AnalysisSummary({ analysis }: { analysis: AssistantAnalysis }): React.ReactElement {
  return (
    <section className="naavre-AIAssistant-analysis" aria-label="Assistant analysis">
      <h3>Notebook generation plan</h3>
      <p>{analysis.summary}</p>
      {analysis.input_schemas?.map(schema => (
        <div key={schema.path} className="naavre-AIAssistant-analysis-group">
          <h4>Verified input columns</h4>
          <p>{schema.path}</p>
          <ul>
            {schema.columns.map(column => (
              <li key={column.name}>
                <code>{column.name}</code>: {column.sample_types.join(', ')}
              </li>
            ))}
          </ul>
          <p>Types estimated from {schema.sample_rows} sampled rows. These are not full-data quality results.</p>
          {schema.warnings.map(warning => <p key={warning}>{warning}</p>)}
        </div>
      ))}
      <SummaryList title="Steps" items={analysis.workflow_steps} />
      <div className="naavre-AIAssistant-analysis-group">
        <h4>Generated cells</h4>
        <ol>
          {analysis.cell_plan.map((cell, index) => (
            <li key={`cell-plan-${index}`}>
              <strong>{cell.title}</strong>: {cell.purpose}
            </li>
          ))}
        </ol>
      </div>
      <SummaryList title="Visualization" items={analysis.visualization_plan} empty="No visualization was identified as necessary." />
      <SummaryList title="Assumptions to review" items={analysis.assumptions} />
      <SummaryList title="Validation checks" items={analysis.validation_requirements} />
    </section>
  );
}

function SummaryList({
  title,
  items,
  empty
}: {
  title: string;
  items: string[];
  empty?: string;
}): React.ReactElement {
  return (
    <div className="naavre-AIAssistant-analysis-group">
      <h4>{title}</h4>
      {items.length ? (
        <ul>
          {items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}
        </ul>
      ) : (
        <p className="naavre-AIAssistant-empty">{empty ?? 'None reported.'}</p>
      )}
    </div>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

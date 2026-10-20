import { ServerConnection } from '@jupyterlab/services';

export type SupportedLanguage = 'python' | 'r';

export interface Lab {
  id: string;
  name: string;
  description: string;
  style: string;
  frameworks: string[];
  document_count: number;
  skill_count: number;
}

export interface KnowledgeItem {
  id?: string;
  filename?: string;
  name?: string;
  title?: string;
  description?: string;
  summary?: string;
  path?: string;
  bytes?: number;
  content?: string;
  text?: string;
  score?: number;
  citation?: { path: string; page?: number | null; hash?: string };
  preview?: string;
  truncated?: boolean;
  [key: string]: unknown;
}

export interface LabKnowledge {
  documents: Array<KnowledgeItem | string>;
  skills: Array<KnowledgeItem | string>;
  hits?: KnowledgeItem[];
  truncated?: boolean;
  [key: string]: unknown;
}

export type AgentStatus =
  | 'queued' | 'running' | 'awaiting_approval'
  | 'completed' | 'failed' | 'cancelled' | 'exhausted';

export interface AgentEvent {
  kind: string;
  message: string;
  tool?: string;
  result?: unknown;
  citation?: string;
}

export interface AgentRun {
  run_id: string;
  lab_id: string;
  status: AgentStatus;
  summary?: string;
  events: AgentEvent[];
  pending_action?: {
    tool: string;
    arguments: unknown;
    [key: string]: unknown;
  } | null;
  context?: Record<string, unknown>;
}

/** Grounded investigation output, carried separately from notebook context. */
export interface AgentContext extends Record<string, unknown> {
  agent_run_id: string;
}

export interface AssistantAnalysis {
  input_schemas?: Array<{
    path: string;
    headers: string[];
    encoding: string;
    delimiter: string;
    sample_rows: number;
    columns: Array<{ name: string; sample_types: string[]; sample_missing: number }>;
    warnings: string[];
    fingerprint: string;
  }>;
  intent: string;
  summary: string;
  relevant_inputs: string[];
  relevant_outputs: string[];
  data_semantics: {
    headers_are_schema_labels: boolean;
    ordinal_targets_data_records_unless_explicit_header: boolean;
    notes: string[];
  };
  workflow_steps: string[];
  visualization_plan: string[];
  assumptions: string[];
  validation_requirements: string[];
  implementation_plan: string[];
  cell_plan: Array<{
    title: string;
    purpose: string;
    inputs: string[];
    outputs: string[];
  }>;
}

export interface GeneratedCell {
  title: string;
  purpose: string;
  code: string;
}

export interface GenerateRequest {
  description: string;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  context?: Record<string, unknown>;
  language: SupportedLanguage;
}

export interface RepairRequest {
  description: string;
  code: string;
  traceback: string;
  language: SupportedLanguage;
  attempt: number;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  context?: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  configured: boolean;
  provider: 'local';
  model: string;
  models?: { agent: string; analysis: string; code: string };
  thinking?: boolean | null;
  schemaContract?: string;
  configurationError?: string | null;
}

interface CellSequenceResponse {
  cells: GeneratedCell[];
  analysis?: AssistantAnalysis;
}

interface CodeResponse {
  code: string;
  analysis?: AssistantAnalysis;
}

interface AnalysisResponse {
  analysis: AssistantAnalysis;
}

export class AssistantClient {
  async labs(signal?: AbortSignal): Promise<Lab[]> {
    const response = await this.request<{ labs: Lab[] }>('labs', { method: 'GET', signal });
    return response.labs;
  }

  async knowledge(labId: string, query = '', signal?: AbortSignal): Promise<LabKnowledge> {
    return this.request<LabKnowledge>(
      `labs/${encodeURIComponent(labId)}/knowledge?q=${encodeURIComponent(query)}`,
      { method: 'GET', signal }
    );
  }

  async uploadDocument(labId: string, filename: string, contentBase64: string): Promise<void> {
    await this.request<unknown>(`labs/${encodeURIComponent(labId)}/documents`, {
      method: 'POST',
      body: JSON.stringify({ filename, content_base64: contentBase64 })
    });
  }

  async startAgentRun(request: {
    lab_id: string;
    description: string;
    context?: Record<string, unknown>;
  }): Promise<Pick<AgentRun, 'run_id' | 'status'>> {
    return this.request<Pick<AgentRun, 'run_id' | 'status'>>('agent/runs', {
      method: 'POST', body: JSON.stringify(request)
    });
  }

  async agentRun(runId: string, signal?: AbortSignal): Promise<AgentRun> {
    return this.request<AgentRun>(`agent/runs/${encodeURIComponent(runId)}`, {
      method: 'GET', signal
    });
  }

  async decideAgentRun(runId: string, approve: boolean): Promise<void> {
    await this.request<unknown>(`agent/runs/${encodeURIComponent(runId)}/decision`, {
      method: 'POST', body: JSON.stringify({ approve })
    });
  }

  async cancelAgentRun(runId: string): Promise<void> {
    await this.request<unknown>(`agent/runs/${encodeURIComponent(runId)}/cancel`, {
      method: 'POST'
    });
  }

  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('health', { method: 'GET' });
  }

  async analyze(
    request: GenerateRequest,
    repair?: Pick<RepairRequest, 'code' | 'traceback' | 'attempt'>
  ): Promise<AssistantAnalysis> {
    const response = await this.request<AnalysisResponse>('analyze', {
      method: 'POST',
      body: JSON.stringify({
        ...request,
        mode: repair ? 'repair' : 'generate',
        ...(repair ?? {})
      })
    });
    return response.analysis;
  }

  async generate(
    request: GenerateRequest,
    analysis: AssistantAnalysis
  ): Promise<GeneratedCell[]> {
    const response = await this.request<CellSequenceResponse>('generate', {
      method: 'POST',
      body: JSON.stringify({ ...request, analysis })
    });
    return response.cells;
  }

  async repair(request: RepairRequest, analysis: AssistantAnalysis): Promise<string> {
    const response = await this.request<CodeResponse>('repair', {
      method: 'POST',
      body: JSON.stringify({ ...request, analysis })
    });
    return response.code;
  }

  private async request<T>(endpoint: string, init: RequestInit): Promise<T> {
    const settings = ServerConnection.makeSettings();
    const baseUrl = settings.baseUrl.endsWith('/')
      ? settings.baseUrl
      : `${settings.baseUrl}/`;
    const response = await ServerConnection.makeRequest(
      `${baseUrl}naavre-ai-assistant/${endpoint}`,
      {
        ...init,
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
          ...init.headers
        }
      },
      settings
    );

    const payload = (await response.json().catch(() => ({}))) as {
      error?: string;
    };
    if (!response.ok) {
      throw new ServerConnection.ResponseError(
        response,
        payload.error ?? `Request failed with status ${response.status}.`
      );
    }
    return payload as T;
  }
}

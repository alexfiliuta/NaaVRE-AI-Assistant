import { ServerConnection } from '@jupyterlab/services';

export type SupportedLanguage = 'python' | 'r';

export interface GenerateRequest {
  description: string;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  language: SupportedLanguage;
}

export interface RepairRequest {
  description: string;
  code: string;
  traceback: string;
  language: SupportedLanguage;
  attempt: number;
}

export interface HealthResponse {
  status: string;
  configured: boolean;
  model: string;
  configurationError?: string | null;
}

interface CodeResponse {
  code: string;
}

export class AssistantClient {
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('health', { method: 'GET' });
  }

  async generate(request: GenerateRequest): Promise<string> {
    const response = await this.request<CodeResponse>('generate', {
      method: 'POST',
      body: JSON.stringify(request)
    });
    return response.code;
  }

  async repair(request: RepairRequest): Promise<string> {
    const response = await this.request<CodeResponse>('repair', {
      method: 'POST',
      body: JSON.stringify(request)
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

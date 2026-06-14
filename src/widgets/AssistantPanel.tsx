import { ReactWidget } from '@jupyterlab/apputils';
import React, { useEffect, useState } from 'react';

import type {
  HealthResponse,
  SupportedLanguage
} from '../api/client';

export interface AssistantSubmission {
  prompt: string;
  language: SupportedLanguage;
  executeAndHeal: boolean;
}

interface AssistantPanelOptions {
  checkHealth: () => Promise<HealthResponse>;
  onSubmit: (submission: AssistantSubmission) => Promise<void>;
}

export class AssistantPanel extends ReactWidget {
  private externalStatus: { message: string; isError: boolean } | null = null;

  constructor(private readonly options: AssistantPanelOptions) {
    super();
    this.addClass('naavre-AIAssistant');
  }

  setStatus(message: string, isError = false): void {
    this.externalStatus = { message, isError };
    this.update();
  }

  protected render(): React.ReactElement {
    return (
      <AssistantView
        {...this.options}
        externalStatus={this.externalStatus}
      />
    );
  }
}

interface AssistantViewProps extends AssistantPanelOptions {
  externalStatus: { message: string; isError: boolean } | null;
}

function AssistantView({
  checkHealth,
  onSubmit,
  externalStatus
}: AssistantViewProps): React.ReactElement {
  const [prompt, setPrompt] = useState('');
  const [language, setLanguage] = useState<SupportedLanguage>('python');
  const [executeAndHeal, setExecuteAndHeal] = useState(false);
  const [isWorking, setIsWorking] = useState(false);
  const [status, setStatus] = useState('Checking server connection...');
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    let disposed = false;
    void checkHealth()
      .then(health => {
        if (!disposed) {
          setIsError(!health.configured);
          setStatus(
            health.configured
              ? `Connected to ${health.model}.`
              : health.configurationError ??
                  'Server connected, but OPENAI_API_KEY is not configured.'
          );
        }
      })
      .catch(error => {
        if (!disposed) {
          setIsError(true);
          setStatus(errorMessage(error));
        }
      });
    return () => {
      disposed = true;
    };
  }, [checkHealth]);

  useEffect(() => {
    if (externalStatus) {
      setStatus(externalStatus.message);
      setIsError(externalStatus.isError);
    }
  }, [externalStatus]);

  const submit = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault();
    if (!prompt.trim() || isWorking) {
      return;
    }
    setIsWorking(true);
    setIsError(false);
    setStatus('Generating code...');
    try {
      await onSubmit({
        prompt: prompt.trim(),
        language,
        executeAndHeal
      });
      setPrompt('');
      setStatus(
        executeAndHeal
          ? 'Generation and self-healing finished. Review the code and outputs.'
          : 'Code inserted in the active notebook for review.'
      );
    } catch (error) {
      setIsError(true);
      setStatus(errorMessage(error));
    } finally {
      setIsWorking(false);
    }
  };

  return (
    <div className="naavre-AIAssistant-content">
      <header className="naavre-AIAssistant-header">
        <h2>NaaVRE AI Assistant</h2>
        <p>Describe one reviewable workflow component.</p>
      </header>

      <form className="naavre-AIAssistant-form" onSubmit={submit}>
        <label htmlFor="naavre-ai-prompt">Component intent</label>
        <textarea
          id="naavre-ai-prompt"
          value={prompt}
          onChange={event => setPrompt(event.target.value)}
          placeholder="Load /tmp/data/input.csv, calculate the mean temperature, and write /tmp/data/summary.csv."
          rows={9}
          disabled={isWorking}
        />

        <fieldset className="naavre-AIAssistant-language">
          <legend>Language</legend>
          <label>
            <input
              type="radio"
              name="naavre-language"
              value="python"
              checked={language === 'python'}
              onChange={() => setLanguage('python')}
              disabled={isWorking}
            />
            Python
          </label>
          <label>
            <input
              type="radio"
              name="naavre-language"
              value="r"
              checked={language === 'r'}
              onChange={() => setLanguage('r')}
              disabled={isWorking}
            />
            R
          </label>
        </fieldset>

        <label className="naavre-AIAssistant-toggle">
          <input
            type="checkbox"
            checked={executeAndHeal}
            onChange={event => setExecuteAndHeal(event.target.checked)}
            disabled={isWorking}
          />
          Execute and repair automatically (maximum 3 repairs)
        </label>

        <button
          type="submit"
          className="jp-mod-styled jp-mod-accept"
          disabled={isWorking || !prompt.trim()}
        >
          {isWorking ? 'Working...' : 'Generate code cell'}
        </button>
      </form>

      <div
        className={`naavre-AIAssistant-status${isError ? ' is-error' : ''}`}
        role="status"
        aria-live="polite"
      >
        {status}
      </div>
    </div>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

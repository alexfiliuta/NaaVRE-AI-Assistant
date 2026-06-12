import { CodeCell } from '@jupyterlab/cells';
import { type NotebookPanel } from '@jupyterlab/notebook';

import {
  AssistantClient,
  type GenerateRequest,
  type SupportedLanguage
} from './api/client';

export interface DraftCellRequest extends GenerateRequest {
  sourceIndex: number;
}

export interface GenerationResult {
  cellIndex: number;
  attempts: number;
  executed: boolean;
  succeeded: boolean;
}

export type ProgressHandler = (message: string) => void;

export function requestFromActiveDraft(
  panel: NotebookPanel,
  fallbackLanguage: SupportedLanguage = 'python'
): DraftCellRequest {
  const notebook = panel.content;
  const cell = notebook.activeCell;
  if (!cell || notebook.activeCellIndex < 0) {
    throw new Error('Select a Markdown draft cell first.');
  }
  if (cell.model.type !== 'markdown') {
    throw new Error('The selected draft must be a Markdown cell.');
  }

  const description = cell.model.sharedModel.getSource().trim();
  if (!description) {
    throw new Error('The selected Markdown cell is empty.');
  }

  const cellMetadata = asRecord(cell.model.sharedModel.getMetadata());
  const naavreMetadata = asRecord(cellMetadata.naavre);
  const language = readLanguage(
    naavreMetadata.language ?? cellMetadata.language,
    fallbackLanguage
  );
  const inputs = asRecord(naavreMetadata.inputs ?? cellMetadata.inputs);
  const outputs = asRecord(naavreMetadata.outputs ?? cellMetadata.outputs);

  return {
    description,
    inputs,
    outputs,
    metadata: { language },
    language,
    sourceIndex: notebook.activeCellIndex
  };
}

export async function generateIntoNotebook(options: {
  panel: NotebookPanel;
  client: AssistantClient;
  request: GenerateRequest;
  insertAfter?: number;
  executeAndHeal: boolean;
  maxRetries: number;
  onProgress?: ProgressHandler;
}): Promise<GenerationResult> {
  const {
    panel,
    client,
    request,
    executeAndHeal,
    maxRetries,
    onProgress = () => undefined
  } = options;
  const notebook = panel.content;
  if (!notebook.model) {
    throw new Error('The active notebook is not ready.');
  }

  onProgress('Generating code...');
  const code = await client.generate(request);
  const currentIndex = notebook.activeCellIndex;
  const insertAfter = options.insertAfter ?? Math.max(currentIndex, -1);
  const cellIndex = insertAfter + 1;
  notebook.model.sharedModel.insertCell(cellIndex, {
    cell_type: 'code',
    source: code,
    metadata: {
      'naavre-ai-assistant': {
        generated: true,
        language: request.language
      }
    }
  });
  notebook.activeCellIndex = cellIndex;
  onProgress('Code inserted for review.');

  if (!executeAndHeal) {
    return {
      cellIndex,
      attempts: 0,
      executed: false,
      succeeded: true
    };
  }

  return executeWithHealing({
    panel,
    client,
    request,
    cellIndex,
    maxRetries,
    onProgress
  });
}

async function executeWithHealing(options: {
  panel: NotebookPanel;
  client: AssistantClient;
  request: GenerateRequest;
  cellIndex: number;
  maxRetries: number;
  onProgress: ProgressHandler;
}): Promise<GenerationResult> {
  const { panel, client, request, cellIndex, maxRetries, onProgress } = options;
  const notebook = panel.content;

  if (!panel.sessionContext.session?.kernel) {
    throw new Error('Start a notebook kernel before using self-healing execution.');
  }
  ensureKernelLanguage(request.language, panel.sessionContext.kernelDisplayName);

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    notebook.activeCellIndex = cellIndex;
    const cell = notebook.widgets[cellIndex];
    if (!(cell instanceof CodeCell)) {
      throw new Error('The generated code cell is no longer available.');
    }

    onProgress(
      attempt === 0
        ? 'Executing generated code...'
        : `Executing repair ${attempt} of ${maxRetries}...`
    );
    await CodeCell.execute(cell, panel.sessionContext);
    const executionError = getExecutionError(cell);
    if (!executionError) {
      onProgress(
        attempt === 0
          ? 'Execution completed successfully.'
          : `Execution succeeded after ${attempt} repair${attempt === 1 ? '' : 's'}.`
      );
      return {
        cellIndex,
        attempts: attempt,
        executed: true,
        succeeded: true
      };
    }

    if (attempt === maxRetries) {
      throw new Error(
        `Execution still fails after ${maxRetries} repair attempts. ` +
          'Review the final traceback in the generated cell.'
      );
    }

    onProgress(`Execution failed. Requesting repair ${attempt + 1}...`);
    const repairedCode = await client.repair({
      description: request.description,
      code: cell.model.sharedModel.getSource(),
      traceback: executionError,
      language: request.language,
      attempt: attempt + 1
    });
    cell.model.sharedModel.setSource(repairedCode);
  }

  throw new Error('The self-healing loop ended unexpectedly.');
}

function getExecutionError(cell: CodeCell): string | null {
  const outputs = cell.model.outputs.toJSON() as Array<Record<string, unknown>>;
  const error = [...outputs]
    .reverse()
    .find(output => output.output_type === 'error');
  if (!error) {
    return null;
  }

  const traceback = Array.isArray(error.traceback)
    ? error.traceback.filter(line => typeof line === 'string').join('\n')
    : '';
  const summary = [error.ename, error.evalue]
    .filter(value => typeof value === 'string' && value)
    .join(': ');
  return traceback || summary || 'The kernel reported an unknown execution error.';
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function readLanguage(
  value: unknown,
  fallback: SupportedLanguage
): SupportedLanguage {
  if (typeof value !== 'string') {
    return fallback;
  }
  return value.toLowerCase() === 'r' ? 'r' : 'python';
}

function ensureKernelLanguage(
  language: SupportedLanguage,
  kernelDisplayName: string
): void {
  const normalized = kernelDisplayName.trim().toLowerCase();
  const matches =
    language === 'python'
      ? normalized.includes('python')
      : /^r(?:\s|$)/.test(normalized);
  if (!matches) {
    throw new Error(
      `Automatic ${language} execution requires a matching kernel; ` +
        `the active kernel is "${kernelDisplayName}". Generate without execution or select another kernel.`
    );
  }
}

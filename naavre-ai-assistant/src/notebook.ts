import { CodeCell } from '@jupyterlab/cells';
import { type NotebookPanel } from '@jupyterlab/notebook';
import { type PartialJSONValue } from '@lumino/coreutils';
import { buildNotebookContext, contextMetadata } from './context';

import {
  AssistantClient,
  type AssistantAnalysis,
  type GeneratedCell,
  type GenerateRequest,
  type SupportedLanguage
} from './api/client';

export interface DraftCellRequest extends GenerateRequest {
  sourceIndex: number;
}

export interface GenerationResult {
  cellIndex: number;
  cellIndices: number[];
  attempts: number;
  executed: boolean;
  succeeded: boolean;
}

export type ProgressHandler = (message: string) => void;
export type AnalysisHandler = (analysis: AssistantAnalysis) => void;

/** Collect bounded, reviewable notebook context without reading arbitrary files. */
export function contextFromNotebook(panel: NotebookPanel): Record<string, unknown> {
  const notebook = panel.content;
  const cells = notebook.widgets.map((cell, index) => ({
    index,
    type: cell.model.type,
    source: cell.model.sharedModel.getSource(),
    metadata: asRecord(cell.model.sharedModel.getMetadata())
  }));
  return buildNotebookContext(panel.context.path, notebook.activeCellIndex, cells);
}

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
    metadata: { language, source_cell_metadata: contextMetadata(cellMetadata) },
    context: contextFromNotebook(panel),
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
  onAnalysis?: AnalysisHandler;
}): Promise<GenerationResult> {
  const {
    panel,
    client,
    request,
    executeAndHeal,
    maxRetries,
    onProgress = () => undefined,
    onAnalysis = () => undefined
  } = options;
  const notebook = panel.content;
  if (!notebook.model) {
    throw new Error('The active notebook is not ready.');
  }

  onProgress('Phase 1/2: understanding the question, metadata, and notebook context...');
  const analysis = await client.analyze(request);
  onAnalysis(analysis);
  onProgress('Phase 2/2: building focused reviewable notebook cells...');
  const cells = await client.generate(request, analysis);
  if (!cells.length) {
    throw new Error('The assistant returned no notebook cells.');
  }
  const currentIndex = notebook.activeCellIndex;
  const insertAfter = options.insertAfter ?? Math.max(currentIndex, -1);
  const cellIndices = cells.map((_, index) => insertAfter + index + 1);
  cells.forEach((cell, index) => {
    notebook.model?.sharedModel.insertCell(cellIndices[index], {
      cell_type: 'code',
      source: cellSource(cell, index, request.language),
      metadata: {
        'naavre-ai-assistant': {
          generated: true,
          sequence: index + 1,
          total: cells.length,
          title: cell.title,
          purpose: cell.purpose,
          language: request.language,
          analysis: analysis as unknown as PartialJSONValue
        }
      }
    });
  });
  notebook.activeCellIndex = cellIndices[0];
  onProgress(`${cells.length} focused code cells inserted for review.`);

  if (!executeAndHeal) {
    return {
      cellIndex: cellIndices[0],
      cellIndices,
      attempts: 0,
      executed: false,
      succeeded: true
    };
  }

  return executeWithHealing({
    panel,
    client,
    request,
    cellIndices,
    maxRetries,
    onProgress,
    onAnalysis
  });
}

async function executeWithHealing(options: {
  panel: NotebookPanel;
  client: AssistantClient;
  request: GenerateRequest;
  cellIndices: number[];
  maxRetries: number;
  onProgress: ProgressHandler;
  onAnalysis: AnalysisHandler;
}): Promise<GenerationResult> {
  const { panel, client, request, cellIndices, maxRetries, onProgress, onAnalysis } = options;
  const notebook = panel.content;

  if (!panel.sessionContext.session?.kernel) {
    throw new Error('Start a notebook kernel before using self-healing execution.');
  }
  ensureKernelLanguage(request.language, panel.sessionContext.kernelDisplayName);

  let totalAttempts = 0;
  for (let cellPosition = 0; cellPosition < cellIndices.length; cellPosition += 1) {
    const cellIndex = cellIndices[cellPosition];
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      notebook.activeCellIndex = cellIndex;
      const cell = notebook.widgets[cellIndex];
      if (!(cell instanceof CodeCell)) {
        throw new Error('The generated code cell is no longer available.');
      }

      onProgress(
        attempt === 0
          ? `Executing cell ${cellPosition + 1} of ${cellIndices.length}...`
          : `Repairing cell ${cellPosition + 1} of ${cellIndices.length} ` +
              `(attempt ${attempt} of ${maxRetries})...`
      );
      await CodeCell.execute(cell, panel.sessionContext);
      const executionError = getExecutionError(cell);
      if (!executionError) {
        totalAttempts += attempt;
        break;
      }

      if (attempt === maxRetries) {
        throw new Error(
          `Cell ${cellPosition + 1} still fails after ${maxRetries} repair attempts. ` +
            'Review the final traceback in the generated cell.'
        );
      }

      onProgress(`Cell ${cellPosition + 1} failed. Requesting repair ${attempt + 1}...`);
      const repairAnalysis = await client.analyze(
        {
          ...request,
          context: request.context
        },
        {
          code: cell.model.sharedModel.getSource(),
          traceback: executionError,
          attempt: attempt + 1
        }
      );
      onAnalysis(repairAnalysis);
      onProgress('Phase 2/2: applying the repair from the validated analysis...');
      const repairedCode = await client.repair(
        {
          description: request.description,
          code: cell.model.sharedModel.getSource(),
          traceback: executionError,
          language: request.language,
          attempt: attempt + 1,
          inputs: request.inputs,
          outputs: request.outputs,
          metadata: request.metadata,
          context: request.context
        },
        repairAnalysis
      );
      cell.model.sharedModel.setSource(repairedCode);
    }
  }

  onProgress(`All ${cellIndices.length} cells executed successfully.`);
  return {
    cellIndex: cellIndices[0],
    cellIndices,
    attempts: totalAttempts,
    executed: true,
    succeeded: true
  };
}

function cellSource(cell: GeneratedCell, index: number, language: SupportedLanguage): string {
  const comment = language === 'r' ? '#' : '#';
  const title = cell.title.replace(/\s+/g, ' ').trim();
  const purpose = cell.purpose.replace(/\s+/g, ' ').trim();
  return `${comment} NaaVRE step ${index + 1}: ${title}\n` +
    `${comment} Purpose: ${purpose}\n\n${cell.code.trim()}`;
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

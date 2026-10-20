import {
  type JupyterFrontEnd,
  type JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  ICommandPalette,
  showErrorMessage,
  ToolbarButton
} from '@jupyterlab/apputils';
import {
  INotebookTracker,
  type NotebookPanel
} from '@jupyterlab/notebook';
import { codeIcon } from '@jupyterlab/ui-components';

import { AssistantClient } from './api/client';
import {
  generateIntoNotebook,
  contextFromNotebook,
  requestFromActiveDraft
} from './notebook';
import {
  AssistantPanel,
  type AssistantSubmission
} from './widgets/AssistantPanel';

const OPEN_COMMAND = 'naavre-ai-assistant:open';
const GENERATE_DRAFT_COMMAND = 'naavre-ai-assistant:generate-from-draft';
const HEAL_DRAFT_COMMAND = 'naavre-ai-assistant:heal-from-draft';

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'naavre-ai-assistant:plugin',
  description: 'Generate and repair NaaVRE workflow cells with an LLM.',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette],
  activate: (
    app: JupyterFrontEnd,
    notebooks: INotebookTracker,
    palette: ICommandPalette | null
  ) => {
    const client = new AssistantClient();
    const panel: AssistantPanel = new AssistantPanel({
      client,
      checkHealth: () => client.health(),
      getEnvironmentContext: (includeNotebook, language) => {
        const notebook = notebooks.currentWidget;
        return boundedInvestigationContext({
          ...(includeNotebook && notebook ? contextFromNotebook(notebook) : {}),
          environment: {
            application: 'jupyterlab',
            path_scope: 'jupyter-server',
            notebook_path: notebook?.context.path ?? null,
            kernel_name: notebook?.sessionContext.session?.kernel?.name ?? null,
            kernel_status: notebook?.sessionContext.session?.kernel?.status ?? null,
            language,
            notebook_context_included: includeNotebook && !!notebook,
            note: 'Paths refer to the Jupyter server, which may be remote. Lab document uploads are knowledge, not dataset uploads. Lab tool actions do not execute generated notebook cells.'
          }
        });
      },
      onSubmit: submission =>
        generateFromPrompt(notebooks, client, submission, panel)
    });
    panel.id = 'naavre-ai-assistant';
    panel.title.caption = 'NaaVRE Research Coworker';
    panel.title.icon = codeIcon;
    panel.title.closable = true;

    app.commands.addCommand(OPEN_COMMAND, {
      label: 'Open NaaVRE AI Assistant',
      icon: codeIcon,
      execute: () => {
        if (!panel.isAttached) {
          app.shell.add(panel, 'left', { rank: 700 });
        }
        app.shell.activateById(panel.id);
      }
    });

    app.commands.addCommand(GENERATE_DRAFT_COMMAND, {
      label: 'Generate Code from Draft Cell',
      icon: codeIcon,
      isEnabled: () => isMarkdownSelected(notebooks.currentWidget),
      execute: () =>
        generateFromDraft(app, notebooks, panel, client, false)
    });

    app.commands.addCommand(HEAL_DRAFT_COMMAND, {
      label: 'Generate, Execute, and Repair Draft Cell',
      icon: codeIcon,
      isEnabled: () => isMarkdownSelected(notebooks.currentWidget),
      execute: () =>
        generateFromDraft(app, notebooks, panel, client, true)
    });

    palette?.addItem({ command: OPEN_COMMAND, category: 'NaaVRE' });
    palette?.addItem({
      command: GENERATE_DRAFT_COMMAND,
      category: 'NaaVRE'
    });
    palette?.addItem({ command: HEAL_DRAFT_COMMAND, category: 'NaaVRE' });
    app.contextMenu.addItem({
      command: GENERATE_DRAFT_COMMAND,
      selector: '.jp-Notebook .jp-MarkdownCell',
      rank: 20
    });
    app.contextMenu.addItem({
      command: HEAL_DRAFT_COMMAND,
      selector: '.jp-Notebook .jp-MarkdownCell',
      rank: 21
    });

    notebooks.widgetAdded.connect((_sender, notebookPanel) => {
      addNotebookToolbarButton(app, notebookPanel);
    });

    void app.commands.execute(OPEN_COMMAND);
  }
};

/** The agent accepts 12 KB of JSON; ordinary notebook generation keeps its existing budget. */
function boundedInvestigationContext(context: Record<string, unknown>): Record<string, unknown> {
  // Python's server-side JSON budget escapes non-ASCII characters (including surrogate pairs).
  const size = (): number => JSON.stringify(context).replace(/[\u007f-\uffff]/g, character =>
    '\\u' + character.charCodeAt(0).toString(16).padStart(4, '0')
  ).length;
  if (size() <= 11500) { return context; }
  const notebook = context.notebook as { active_cell_index?: number } | undefined;
  const activeIndex = notebook?.active_cell_index ?? 0;
  const cells = Array.isArray(context.cells)
    ? [...context.cells] as Array<{ index: number; source: string; source_truncated?: boolean }>
    : [];
  // Keep cells closest to the active cell for as long as the budget permits.
  cells.sort((a, b) => Math.abs(a.index - activeIndex) - Math.abs(b.index - activeIndex));
  context.cells = cells;
  context.context_selection = { truncated: true, note: 'Selected context bounded for lab investigation.' };
  while (size() > 11500 && cells.length) {
    if (cells.length === 1 && cells[0].source.length > 200) {
      cells[0] = { ...cells[0], source: cells[0].source.slice(0, Math.floor(cells[0].source.length / 2)), source_truncated: true };
    } else {
      cells.pop();
    }
  }
  cells.sort((a, b) => a.index - b.index);
  if (size() > 11500) { delete context.available_files; }
  if (size() > 11500) { delete context.notebook; }
  if (size() > 11500) {
    const environment = context.environment as Record<string, unknown>;
    context.environment = { ...environment, notebook_path: String(environment.notebook_path ?? '').slice(0, 500) };
  }
  return context;
}

async function generateFromDraft(
  app: JupyterFrontEnd,
  notebooks: INotebookTracker,
  assistantPanel: AssistantPanel,
  client: AssistantClient,
  executeAndHeal: boolean
): Promise<void> {
  await app.commands.execute(OPEN_COMMAND);
  assistantPanel.setStatus('Reading selected draft cell...');
  try {
    const notebookPanel = requireNotebook(notebooks.currentWidget);
    const request = requestFromActiveDraft(notebookPanel);
    await generateIntoNotebook({
      panel: notebookPanel,
      client,
      request,
      insertAfter: request.sourceIndex,
      executeAndHeal,
      maxRetries: 3,
      onProgress: message => assistantPanel.setStatus(message),
      onAnalysis: analysis => assistantPanel.setAnalysis(analysis)
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    assistantPanel.setStatus(message, true);
    await showErrorMessage('NaaVRE AI Assistant', message);
  }
}

async function generateFromPrompt(
  notebooks: INotebookTracker,
  client: AssistantClient,
  submission: AssistantSubmission,
  assistantPanel: AssistantPanel
): Promise<void> {
  const panel = requireNotebook(notebooks.currentWidget);
  const agentContext = submission.agentContext;
  assistantPanel.setAnalysis(null);
  await generateIntoNotebook({
    panel,
    client,
    request: {
      description: submission.prompt,
      language: submission.language,
      metadata: {
        visualization_requested: submission.generateVisualization,
        context_source: submission.includeNotebookContext ? 'active-notebook' : 'none',
        ...(submission.labId ? { lab_id: submission.labId } : {}),
        ...(agentContext ? { agent_run_id: agentContext.agent_run_id } : {})
      },
      context: {
        ...(submission.includeNotebookContext ? contextFromNotebook(panel) : {}),
        // The backend resolves this reference to owned, completed evidence. Browser
        // copies of tool results are for display and must not become generation facts.
        ...(agentContext ? { agent_run_id: agentContext.agent_run_id } : {})
      }
    },
    executeAndHeal: submission.executeAndHeal,
    maxRetries: 3,
    onAnalysis: analysis => assistantPanel.setAnalysis(analysis),
    onProgress: message => assistantPanel.setStatus(message)
  });
}

function addNotebookToolbarButton(
  app: JupyterFrontEnd,
  panel: NotebookPanel
): void {
  const button = new ToolbarButton({
    icon: codeIcon,
    tooltip: 'Generate code from the selected Markdown draft cell',
    onClick: () => {
      void app.commands.execute(GENERATE_DRAFT_COMMAND);
    }
  });
  panel.toolbar.insertItem(10, 'naavre-ai-generate-draft', button);
}

function isMarkdownSelected(panel: NotebookPanel | null): boolean {
  return panel?.content.activeCell?.model.type === 'markdown';
}

function requireNotebook(panel: NotebookPanel | null): NotebookPanel {
  if (!panel) {
    throw new Error('Open a notebook before generating code.');
  }
  return panel;
}

export default plugin;

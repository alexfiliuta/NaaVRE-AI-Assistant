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
    const panel = new AssistantPanel({
      checkHealth: () => client.health(),
      onSubmit: submission =>
        generateFromPrompt(notebooks, client, submission)
    });
    panel.id = 'naavre-ai-assistant';
    panel.title.caption = 'NaaVRE AI Assistant';
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
      onProgress: message => assistantPanel.setStatus(message)
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
  submission: AssistantSubmission
): Promise<void> {
  const panel = requireNotebook(notebooks.currentWidget);
  await generateIntoNotebook({
    panel,
    client,
    request: {
      description: submission.prompt,
      language: submission.language
    },
    executeAndHeal: submission.executeAndHeal,
    maxRetries: 3
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

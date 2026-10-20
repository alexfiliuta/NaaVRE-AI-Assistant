export interface NotebookContextCell {
  index: number;
  type: string;
  source: string;
  metadata: Record<string, unknown>;
  source_truncated?: boolean;
}

const CONTEXT_BUDGET = 32000;
const MAX_CONTEXT_CELLS = 40;

/** Stored plans remain in the notebook, but are not repeated in every model request. */
export function contextMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const result = { ...metadata };
  const assistant = result['naavre-ai-assistant'];
  if (assistant && typeof assistant === 'object' && !Array.isArray(assistant)) {
    const { analysis: _analysis, ...summary } = assistant as Record<string, unknown>;
    result['naavre-ai-assistant'] = summary;
  }
  return result;
}

export function buildNotebookContext(
  path: string,
  activeIndex: number,
  cells: NotebookContextCell[]
): Record<string, unknown> {
  let truncated = false;
  const selected: NotebookContextCell[] = [];
  const availableFiles: unknown[] = [];
  const fileKeys = new Set<string>();
  let filesLength = 0;
  for (const cell of cells) {
    const nested = cell.metadata.naavre;
    const naavre = nested && typeof nested === 'object' ? nested as Record<string, unknown> : {};
    const files = naavre.files ?? cell.metadata.files;
    if (!Array.isArray(files)) {
      continue;
    }
    for (const file of files) {
      const key = JSON.stringify(file);
      if (fileKeys.has(key)) {
        continue;
      }
      fileKeys.add(key);
      if (filesLength + key.length > 4000) {
        truncated = true;
        continue;
      }
      availableFiles.push(file);
      filesLength += key.length;
    }
  }
  const result = {
    notebook: { path: path.slice(0, 2000), cell_count: cells.length, active_cell_index: activeIndex },
    cells: selected,
    available_files: availableFiles,
    context_selection: { truncated: true },
    note: 'Selected notebook context only; omitted cells may define dependencies. Verified input schemas are supplied separately by the server.'
  };
  const priority = [...cells].sort((a, b) =>
    Math.abs(a.index - activeIndex) - Math.abs(b.index - activeIndex) || a.index - b.index
  );
  for (const original of priority.slice(0, MAX_CONTEXT_CELLS)) {
    let metadata = contextMetadata(original.metadata);
    const metadataText = JSON.stringify(metadata);
    if (metadataText.length > 4000) {
      metadata = { preview: metadataText.slice(0, 2000), truncated: true };
      truncated = true;
    }
    const cell = { ...original, source: original.source.slice(0, 6000), metadata };
    cell.source_truncated = cell.source.length < original.source.length;
    selected.push(cell);
    while (JSON.stringify(result).length > CONTEXT_BUDGET && cell.source.length > 200) {
      cell.source = cell.source.slice(0, Math.floor(cell.source.length / 2));
      cell.source_truncated = true;
    }
    if (JSON.stringify(result).length > CONTEXT_BUDGET) {
      selected.pop();
    }
    truncated ||= cell.source_truncated;
  }
  selected.sort((a, b) => a.index - b.index);
  result.context_selection.truncated = truncated || selected.length < cells.length;
  return result;
}

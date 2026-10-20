const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const application = path.resolve(__dirname, '../../naavre-ai-assistant');
const ts = Module.createRequire(path.join(application, 'package.json'))('typescript');

const sourcePath = path.join(application, 'src/context.ts');
const compiled = ts.transpileModule(fs.readFileSync(sourcePath, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
});
const contextModule = new Module(sourcePath, module);
contextModule._compile(compiled.outputText, sourcePath);
const { buildNotebookContext } = contextModule.exports;

test('large notebook is bounded, preserves active cell, removes copied plans without mutation', () => {
  const cells = Array.from({length: 80}, (_, index) => ({
    index, type: 'code', source: '# 温度 \\ \n'.repeat(4000),
    metadata: {
      naavre: {inputs: {'observations.csv': 'temperature'}, files: ['observations.csv']},
      'naavre-ai-assistant': {title: `Job ${index}`, analysis: {summary: 'COPIED_PLAN'.repeat(1000)}}
    }
  }));
  const before = JSON.stringify(cells);
  const result = buildNotebookContext('Untitled7.ipynb', 65, cells);
  assert.ok(JSON.stringify(result).length <= 32000);
  assert.ok(result.cells.some(cell => cell.index === 65));
  assert.deepEqual(result.cells.map(c => c.index), result.cells.map(c => c.index).sort((a,b) => a-b));
  assert.equal(result.context_selection.truncated, true);
  assert.deepEqual(result.available_files, ['observations.csv']);
  assert.ok(!JSON.stringify(result).includes('COPIED_PLAN'));
  assert.equal(JSON.stringify(cells), before);
});

test('small notebooks retain their source and scientific metadata', () => {
  const result = buildNotebookContext('test.ipynb', 0, [{
    index: 0, type: 'code', source: 'x = 1', metadata: {units: 'Celsius'}
  }]);
  assert.equal(result.cells[0].source, 'x = 1');
  assert.deepEqual(result.cells[0].metadata, {units: 'Celsius'});
  assert.equal(result.context_selection.truncated, false);
});

test('large metadata and file declarations cannot overflow the request budget', () => {
  const result = buildNotebookContext('test.ipynb', 0, [{
    index: 0, type: 'code', source: 'x = 1', metadata: {
      data: 'x'.repeat(100000), files: ['y'.repeat(100000), 'observations.csv']
    }
  }]);
  assert.ok(JSON.stringify(result).length <= 32000);
  assert.equal(result.context_selection.truncated, true);
  assert.deepEqual(result.available_files, ['observations.csv']);
});

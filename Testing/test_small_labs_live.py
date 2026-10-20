"""Four real model-led investigations/builds. Generation only; source review before execution.

Calls the same AgentRuntime and LLMCodeGenerator as the application; not a browser UI test.
Never forces MCP calls or injects reference answers. Original labs/notebooks are unchanged.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "naavre-ai-assistant"))
import nbformat
from naavre_ai_assistant.agent_runtime import AgentLimits, AgentRuntime
from naavre_ai_assistant.labs import LabRegistry
from naavre_ai_assistant.llm_service import LLMConfig
from compare_energy_models import MeasuredGenerator
from small_lab_cases import ROOT, CASES, DATA, PROMPTS, CONTRACTS, TOOLS, make_lab, digest


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


async def main(args):
    batch = ROOT / 'Testing/results/small-lab-tests' / args.batch
    batch.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout),
        logging.FileHandler(batch / 'generation.log', encoding='utf-8')])
    with urlopen('http://127.0.0.1:11434/api/tags', timeout=15) as response:
        models = json.load(response)['models']
    model = next(m for m in models if m['name'] == args.model)
    save(batch / 'model.json', model)
    failures = 0
    for case in CASES:
        folder = batch / case['id']
        folder.mkdir()
        lab_root = make_lab(folder / 'labs', case['lab'], case['knowledge'], case['skill'])
        os.environ['NAAVRE_LABS_DIR'] = str(lab_root.parent)
        os.environ['NAAVRE_ALLOWED_LABS'] = 'case'
        source = lab_root / 'data' / DATA[case['lab']][0]
        request = PROMPTS[case['lab']][case['prompt']] + ' Do not modify the input. Save derived results.'
        metrics = {**case, 'model': args.model, 'model_digest': model['digest'],
            'started_utc': datetime.now(timezone.utc).isoformat(), 'request': request,
            'input_sha256': digest(source), 'knowledge_sha256': digest(lab_root/'knowledge/protocol.md'),
            'skill_sha256': digest(lab_root/'skills/analysis-procedure.md'), 'context_window': 16384,
            'thinking': False, 'timeout_seconds': None, 'max_tokens': None,
            'mode': 'normal model-led backend workflow, not a browser UI test', 'status': 'investigating'}
        save(folder / 'metrics.json', metrics)
        service = MeasuredGenerator(LLMConfig(base_url='http://127.0.0.1:11434/v1', model=args.model,
            timeout_seconds=None, max_tokens=None, context_window=16384, thinking=False), folder)
        began = time.monotonic()
        try:
            state_dir = ROOT / 'Testing/results/st' / uuid.uuid4().hex[:8]
            metrics['state_dir'] = str(state_dir)
            manager = AgentRuntime(LabRegistry(), state_dir, service=service,
                allowed_tools=TOOLS, limits=AgentLimits(max_steps=12, run_seconds=None, operation_seconds=None))
            run_id = manager.start('small-lab-evaluation', 'case', request)['run_id']
            await manager._runs[run_id].task
            snapshot = manager.get('small-lab-evaluation', run_id)
            save(folder / 'investigation.json', snapshot)
            metrics['investigation_seconds'] = time.monotonic() - began
            if snapshot['status'] != 'completed':
                raise RuntimeError('Investigation ended as ' + snapshot['status'])
            context = manager.verified_context('small-lab-evaluation', run_id)
            save(folder / 'context.json', context)
            build_request = request + '\nInput: ' + source.as_posix() + '. Write only under ' + (folder / 'outputs').as_posix() + '.\n' + CONTRACTS[case['lab']]
            metrics['build_request'] = build_request
            metrics['status'] = 'planning'
            save(folder / 'metrics.json', metrics)
            start = time.monotonic()
            analysis = await asyncio.to_thread(service.analyze_request, build_request, context=context)
            save(folder / 'analysis.json', analysis)
            metrics['planning_seconds'] = time.monotonic() - start
            metrics['status'] = 'generating'
            save(folder / 'metrics.json', metrics)
            start = time.monotonic()
            cells = await asyncio.to_thread(service.generate_cells_from_analysis, build_request,
                analysis=analysis, context=context)
            save(folder / 'cells.json', cells)
            notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
                '# Isolated test: ' + case['id'] + '\n\n' + request + '\n\nUnedited model cells. No expected answers provided.')])
            for cell in cells:
                notebook.cells.extend([nbformat.v4.new_markdown_cell(cell['title'] + '\n\n' + cell['purpose']),
                                      nbformat.v4.new_code_cell(cell['code'])])
            notebook.metadata.kernelspec = {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'}
            nbformat.write(notebook, folder / 'generated.ipynb')
            metrics.update(status='generated_pending_source_review', generated_cells=len(cells),
                code_generation_seconds=time.monotonic()-start, notebook_sha256=digest(folder/'generated.ipynb'))
        except Exception as exc:
            failures += 1
            metrics.update(failed_phase=metrics['status'], status='failed', error=str(exc))
            logging.exception('Live test failed: %s', case['id'])
        finally:
            metrics.update(total_seconds=time.monotonic()-began, input_unchanged=digest(source)==metrics['input_sha256'])
            save(folder / 'metrics.json', metrics)
            print(json.dumps(metrics, indent=2), flush=True)
    print('All four live generation attempts recorded. Execution requires source review.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3.8:27b')
    parser.add_argument('--batch', default=datetime.now().strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    if not args.batch or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.batch):
        parser.error('Batch must be a simple identifier')
    sys.exit(asyncio.run(main(args)))

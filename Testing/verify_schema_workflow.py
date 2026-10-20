"""Opt-in live Qwen verification; generate first, review the artifact, then execute.

Usage from the workspace: python Testing/verify_schema_workflow.py generate|execute
Test outputs are restricted to Testing/results/schema-verification; the source CSV is read-only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from fixture_paths import LABS_ROOT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'naavre-ai-assistant'))
from naavre_ai_assistant.llm_service import LLMCodeGenerator

INPUT = LABS_ROOT / 'environment' / 'data' / 'station_observations.csv'
OUTPUT = ROOT / 'Testing' / 'results' / 'schema-verification'
ARTIFACT = OUTPUT / 'generated.json'


class ReportingGenerator(LLMCodeGenerator):
    def _request_content(self, *args, **kwargs):
        print('Request:', kwargs.get('response_kind', 'code'), flush=True)
        content = super()._request_content(*args, **kwargs)
        self._test_request_count = getattr(self, '_test_request_count', 0) + 1
        (OUTPUT / f'response-{self._test_request_count:02}.txt').write_text(content, encoding='utf-8')
        print('Received:', kwargs.get('response_kind', 'code'), flush=True)
        return content


def generate():
    with INPUT.open(encoding='utf-8-sig', newline='') as stream:
        headers = next(csv.reader(stream))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt = (
        f'My input CSV is {INPUT.as_posix()}. Do not modify the source file. '
        'Assess its quality, find unusual temperature measurements, compare stations over time, '
        f'create a useful visualization, and save derived results under {OUTPUT.as_posix()}. '
        f'The columns are {", ".join(headers)}.'
    )
    service = ReportingGenerator()
    started = time.monotonic()
    analysis = service.analyze_request(prompt)
    print('Verified headers:', analysis['input_schemas'][0]['headers'], flush=True)
    print('Planned cells:', [item['title'] for item in analysis['cell_plan']], flush=True)
    cells = service.generate_cells_from_analysis(prompt, analysis=analysis)
    artifact = dict(prompt=prompt, analysis=analysis, cells=cells)
    ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print('Generated', len(cells), 'cells in', round(time.monotonic()-started, 1), 'seconds', flush=True)
    print('Review:', ARTIFACT, flush=True)


def execute():
    artifact = json.loads(ARTIFACT.read_text(encoding='utf-8'))
    original = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    os.chdir(OUTPUT)

    def check_write(target):
        if isinstance(target, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(target)).resolve()
            if not resolved.is_relative_to(OUTPUT.resolve()):
                raise PermissionError(f'Test forbids writes outside {OUTPUT}: {resolved}')

    def audit(event, args):
        if event == 'open':
            target, mode, flags = args
            if (mode and any(char in mode for char in 'wax+')) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                check_write(target)
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
            check_write(args[0])
        if event in ('os.rename', 'os.link', 'os.symlink'):
            check_write(args[0]); check_write(args[1])
        if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.exec', 'os.fork'):
            raise PermissionError(f'External process/network access forbidden during test: {event}')

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    scope = {'__name__': '__main__'}
    for index, cell in enumerate(artifact['cells'], 1):
        print('Executing', index, cell['title'], flush=True)
        exec(compile(cell['code'], f'<generated cell {index}>', 'exec'), scope)
        print('PASS', index, flush=True)
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == original
    print('Source CSV unchanged. Output files:', [item.name for item in OUTPUT.iterdir() if item.is_file()], flush=True)


def main():
    global INPUT, OUTPUT, ARTIFACT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generate', 'execute'])
    parser.add_argument('--input', type=Path, default=INPUT,
                        help='CSV to inspect; defaults to the bundled environment lab fixture')
    parser.add_argument('--output', type=Path, default=OUTPUT,
                        help='Directory for generated.json and derived results')
    args = parser.parse_args()
    INPUT = args.input.resolve(strict=True)
    OUTPUT = args.output.resolve()
    ARTIFACT = OUTPUT / 'generated.json'
    {'generate': generate, 'execute': execute}[args.stage]()


if __name__ == '__main__':
    main()

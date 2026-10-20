"""Relocation must keep frozen source intact and still check the real data."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from portable_paths import reviewed_input_mapping, path_mapping_source
from small_lab_reference import reviewed_cells, SOURCES


@pytest.mark.parametrize('kind', ['plants', 'energy'])
def test_frozen_loading_cell_reads_current_fixture_without_editing_code(kind, tmp_path):
    sources = reviewed_cells(kind)
    original = list(sources)
    mapping = reviewed_input_mapping(sources, SOURCES[kind][0])
    # A subprocess confines the pathlib adaptation to the verification kernel.
    code = path_mapping_source(mapping) + '\n' + sources[0] + (
        '\nimport json\nprint(json.dumps({"rows": len(raw_records), '
        '"path_type_preserved": isinstance(input_path, Path)}))\n'
    )
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    answer = json.loads(result.stdout.splitlines()[-1])
    assert answer == {'rows': 24 if kind == 'plants' else 28, 'path_type_preserved': True}
    assert sources == original


def test_runtime_mapping_leaves_other_paths_and_write_auditing_intact(tmp_path):
    actual = tmp_path / 'input.csv'
    actual.write_text('value\n7\n')
    legacy = 'C:/old-machine/input.csv'
    other = tmp_path / 'unmapped.txt'
    code = path_mapping_source({legacy: str(actual)}) + f'''
import sys
from pathlib import Path
def audit(event, args):
    if event == 'open' and args[1] == 'w' and str(args[0]) == {str(actual)!r}:
        raise PermissionError('Actual relocated input remains protected')
sys.addaudithook(audit)
assert Path({legacy!r}).is_file()
assert Path({legacy!r}).read_text() == 'value\\n7\\n'
Path({str(other)!r}).write_text('other')
try:
    Path({legacy!r}).write_text('changed')
except PermissionError:
    pass
else:
    raise AssertionError('Write audit was bypassed')
'''
    subprocess.run([sys.executable, '-c', code], check=True, cwd=tmp_path)
    assert actual.read_text() == 'value\n7\n'
    assert other.read_text() == 'other'


def test_unexpected_loading_declaration_is_rejected(tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('value\n')
    with pytest.raises(ValueError, match='exactly one literal'):
        reviewed_input_mapping(['input_path = choose_a_path()'], source)
    with pytest.raises(ValueError, match='same reviewed dataset filename'):
        reviewed_input_mapping(['input_path = Path("C:/old/different.csv")'], source)


def test_walkthrough_sections_match_relative_links_and_legacy_inventory():
    workspace = Path(__file__).resolve().parents[2]
    checker_path = workspace / 'Learning/code-walkthrough-2026-10-10/verify_walkthrough.py'
    if not checker_path.is_file():
        pytest.skip('The optional personal walkthrough is not included in the code repository.')
    spec = importlib.util.spec_from_file_location('walkthrough_path_checker', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    identity = 'naavre-ai-assistant/naavre_ai_assistant/labs.py'
    relative = '../../' + identity
    historical = 'C:/old-machine/project/' + identity
    assert checker.source_identity(relative) == identity
    assert checker.source_identity(historical) == identity
    document = f'Source: [{relative}]({relative})\nLabRegistry details\nSource: [other](other)\nOther details'
    section = checker.section_for(identity, document)
    assert 'LabRegistry details' in section
    assert 'Other details' not in section


@pytest.mark.parametrize('local_labs', [False, True], ids=['public-fixtures', 'local-labs'])
def test_relocated_launcher_selects_its_own_environment_and_clears_stale_configuration(tmp_path, monkeypatch, local_labs):
    workspace = Path(__file__).resolve().parents[2]
    relocated = tmp_path / 'Relocated project with spaces'
    scripts = relocated / 'naavre-ai-assistant/scripts'
    scripts.mkdir(parents=True)
    script = scripts / 'start_demo.py'
    script.write_bytes((workspace / 'naavre-ai-assistant/scripts/start_demo.py').read_bytes())
    labs = relocated / ('Labs' if local_labs else 'Testing/fixtures/labs')
    labs.mkdir(parents=True)
    python = relocated / 'naavre-ai-assistant/.venv' / (
        'Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.touch()
    spec = importlib.util.spec_from_file_location('relocated_demo_launcher', script)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    elsewhere = tmp_path / 'Different caller directory'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    stale = {
        'PYTHONHOME': 'obsolete-python-home', 'PYTHONPATH': 'obsolete-checkout',
        'VIRTUAL_ENV': 'obsolete-venv', 'NAAVRE_ENV_FILE': 'obsolete.env',
        'NAAVRE_AGENT_STATE_DIR': 'obsolete-state',
        'LOCAL_LLM_REMOTE_HOST': 'old-server.example',
        'LOCAL_LLM_API_KEY_FILE': 'obsolete.token',
        'LOCAL_LLM_CLIENT_CERT': 'obsolete.pem',
        'LOCAL_LLM_CLIENT_KEY': 'obsolete.key',
        'LOCAL_LLM_CA_FILE': 'obsolete-ca.pem',
        'LOCAL_LLM_CODE_MODEL': 'obsolete-model',
    }
    for key, value in stale.items():
        monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs:
                        calls.append((args, kwargs)) or SimpleNamespace(returncode=0))
    assert launcher.main(['--check', '--labs', 'plants,energy']) == 0
    args, kwargs = calls[0]
    assert args[0] == [str(python), '-m', 'naavre_ai_assistant', 'check']
    assert kwargs['cwd'] == relocated
    env = kwargs['env']
    assert env['NAAVRE_LABS_DIR'] == str(labs)
    assert env['NAAVRE_ALLOWED_LABS'] == 'plants,energy'
    assert env['LOCAL_LLM_BASE_URL'] == 'http://127.0.0.1:11434/v1'
    assert env['VIRTUAL_ENV'] == str(relocated / 'naavre-ai-assistant/.venv')
    assert all(key not in env for key in stale if key != 'VIRTUAL_ENV')
    # Build the child environment without modifying the caller's settings.
    assert os.environ['PYTHONPATH'] == 'obsolete-checkout'

import json
import importlib.util
from pathlib import Path
import zipfile

import pytest

from naavre_ai_assistant.__main__ import check, init_lab
from naavre_ai_assistant.labs import LabRegistry
from fixture_paths import LABS_ROOT


def test_init_empty_lab_has_real_tool_allowlist(tmp_path, monkeypatch):
    monkeypatch.delenv('NAAVRE_ALLOWED_LABS', raising=False)
    target = init_lab(tmp_path, 'study', 'My study')
    lab = LabRegistry(tmp_path).get('study')
    assert lab.name == 'My study'
    assert len(lab.allowed_tools) == 4
    assert lab.files() == [] and lab.skills() == []
    assert not list(target.rglob('*.md'))
    with pytest.raises(FileExistsError):
        init_lab(tmp_path, 'study', 'Must not overwrite')


@pytest.mark.parametrize('identifier', ['../study', 'Study', '', 'a/b'])
def test_invalid_lab_id_creates_nothing(tmp_path, identifier):
    with pytest.raises(ValueError):
        init_lab(tmp_path, identifier, 'Study')
    assert list(tmp_path.iterdir()) == []


def test_cli_diagnostic_has_no_secret_fields(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv('NAAVRE_LABS_DIR', str(tmp_path))
    monkeypatch.setenv('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:11434/v1')
    monkeypatch.delenv('LOCAL_LLM_API_KEY_FILE', raising=False)
    assert check() == 0
    value = json.loads(capsys.readouterr().out)
    assert value['backend'] == 'Jupyter Server / Tornado'
    assert 'token_file' not in value and 'client_key' not in value


def test_source_checkout_uses_external_labs_directory(monkeypatch):
    import naavre_ai_assistant.labs as lab_module
    monkeypatch.delenv('NAAVRE_LABS_DIR', raising=False)
    application = Path(lab_module.__file__).resolve().parents[1]
    assert LabRegistry().root == LABS_ROOT
    assert not (application / 'labs').exists()


def test_windows_build_metadata_is_portable_to_linux(tmp_path):
    workspace = Path(__file__).resolve().parents[2]
    script = workspace / 'naavre-ai-assistant/scripts/build_frontend.py'
    if not script.is_file():
        script = workspace / 'Operations/build_frontend.py'
    spec = importlib.util.spec_from_file_location('build_frontend', script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = tmp_path/'package.json'
    path.write_text(json.dumps({'jupyterlab': {'_build': {'load': 'static\\remoteEntry.js'}}}))
    builder.normalize_bundle(path)
    assert json.loads(path.read_text())['jupyterlab']['_build']['load'] == 'static/remoteEntry.js'


@pytest.mark.parametrize('artifact', ['outputs/plot.svg', 'reports/report.md',
                                     'labs/study/data.csv', 'learning/practice.py'])
def test_wheel_check_rejects_leftover_artifacts_before_installing(tmp_path, monkeypatch, artifact):
    spec = importlib.util.spec_from_file_location('verify_package', Path(__file__).resolve().parents[1]/'verify_package.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    wheel = tmp_path/'package.whl'
    frontend = {'jupyterlab': {'_build': {'load': 'static/remoteEntry.js'},
                'sharedPackages': {'@jupyterlab/apputils': {'requiredVersion': '^4.2.0'}}}}
    with zipfile.ZipFile(wheel, 'w') as bundle:
        bundle.writestr('naavre_ai_assistant/labextension/package.json', json.dumps(frontend))
        bundle.writestr('release.data/data/share/jupyter/labextensions/naavre-ai-assistant/static/remoteEntry.js', '')
        bundle.writestr('release.data/data/etc/jupyter/jupyter_server_config.d/assistant.json', '{}')
        bundle.writestr('release.dist-info/'+artifact, 'must not ship')
    calls = []
    monkeypatch.setattr(verifier.subprocess, 'run', lambda *a, **kw: calls.append(a))
    with pytest.raises(AssertionError):
        verifier.verify(wheel)
    assert not calls

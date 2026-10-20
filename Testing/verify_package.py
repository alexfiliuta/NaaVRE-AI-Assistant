"""Install a wheel outside the checkout and smoke-test authenticated JupyterHub-style routes.

Uses the current Python environment's dependencies, but imports this extension
only from the installed wheel. Does not contact Ollama or change an existing server.
"""
import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'naavre-ai-assistant'))


def verify(wheel: Path) -> None:
    wheel = wheel.resolve(strict=True)
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        assert any('share/jupyter/labextensions/naavre-ai-assistant/static/remoteEntry.' in name for name in names)
        assert any('etc/jupyter/jupyter_server_config.d/' in name for name in names)
        frontend = json.loads(bundle.read('naavre_ai_assistant/labextension/package.json'))
        assert '\\' not in frontend['jupyterlab']['_build']['load']
        assert frontend['jupyterlab']['sharedPackages']['@jupyterlab/apputils']['requiredVersion'] == '^4.2.0'
        assert not any(name.endswith(('.env', '.ipynb', '.csv', '.tsv', '.svg', '.md', '.pem', '.key', '.map')) for name in names)
        assert not any(part in name.split('/') for name in names for part in (
            'node_modules', '.venv', 'test-results', 'docs', 'reports', 'output', 'outputs', 'labs', 'learning', 'tests'))
        metadata_paths = [name for name in names if name.count('/') == 1
                          and name.endswith('.dist-info/METADATA')]
        assert len(metadata_paths) == 1, 'Wheel must contain one distribution metadata record'
        expected_version = BytesParser().parsebytes(bundle.read(metadata_paths[0])).get('Version')
        assert expected_version, 'Wheel metadata is missing its version'
    with tempfile.TemporaryDirectory(prefix='naavre-wheel-') as folder:
        temporary = Path(folder)
        target = temporary/'installed'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile',
                        '--target', str(target), str(wheel)], check=True)
        # -I and a foreign working directory stop checkout/PYTHONPATH imports.
        code = ("import sys;sys.path.insert(0," + repr(str(target)) + ");"
                "import naavre_ai_assistant as p;print(p.__file__);"
                "assert p.__version__==" + repr(expected_version) + ";"
                "assert p.__file__.startswith(" + repr(str(target)) + ")")
        subprocess.run([sys.executable, '-I', '-c', code], cwd=temporary, check=True)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('NAAVRE_', 'LOCAL_LLM_'))}
        env.update(NAAVRE_LABS_DIR=str(temporary/'labs'), NAAVRE_AGENT_STATE_DIR=str(temporary/'state'),
                   LOCAL_LLM_BASE_URL='http://127.0.0.1:11434/v1', LOCAL_LLM_MODEL='qwen3-coder:30b',
                   JUPYTER_CONFIG_PATH=str(target/'etc/jupyter'),
                   JUPYTER_RUNTIME_DIR=str(temporary/'runtime'),
                   JUPYTER_PATH=str(target/'share/jupyter'))
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        prefix = '/user/wheel-test/'
        token = 'packaging-test-only-token'
        argv = ['--no-browser', '--ip=127.0.0.1', f'--port={port}', '--ServerApp.port_retries=0',
                '--ServerApp.open_browser=False', f'--ServerApp.root_dir={temporary}',
                f'--ServerApp.base_url={prefix}', f'--IdentityProvider.token={token}',
                '--LabApp.extension_manager=readonly']
        startup = ("import sys;sys.path.insert(0," + repr(str(target)) + ");"
                   "from jupyterlab.labapp import LabApp;LabApp.launch_instance(" + repr(argv) + ")")
        with (temporary/'server.log').open('w+', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-I', '-c', startup], cwd=temporary,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                from naavre_ai_assistant.llm_transport import open_request
                base = f'http://127.0.0.1:{port}{prefix}naavre-ai-assistant/'
                deadline = time.monotonic()+60
                health = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        with open_request(Request(base+'health', headers={'Authorization': 'token '+token}), timeout=2) as response:
                            health = json.load(response)
                        break
                    except OSError:
                        time.sleep(.2)
                if health is None:
                    log.seek(0)
                    raise AssertionError('Wheel server did not start:\n'+log.read()[-6000:])
                assert health['configured'] is True
                try:
                    open_request(Request(base+'health'), timeout=5)
                except HTTPError as error:
                    assert error.code in (302, 403)
                else:
                    raise AssertionError('Unauthenticated request unexpectedly succeeded')
                with open_request(Request(base+'labs', headers={'Authorization': 'token '+token}), timeout=5) as response:
                    assert json.load(response) == {'labs': []}
                try:
                    open_request(Request(base+'analyze', data=b'{}', headers={
                        'Authorization': 'token '+token, 'Content-Type': 'application/json'}), timeout=5)
                except HTTPError as error:
                    assert error.code == 400
                else:
                    raise AssertionError('Invalid analysis request unexpectedly succeeded')
                asset = next(name.split('/share/jupyter/labextensions/')[1] for name in names
                             if '/share/jupyter/labextensions/' in name and '/static/remoteEntry.' in name)
                url = f'http://127.0.0.1:{port}{prefix}lab/extensions/{asset}'
                with open_request(Request(url, headers={'Authorization': 'token '+token}), timeout=5) as response:
                    assert response.status == 200 and len(response.read()) > 100
                print('PASS: installed wheel imports, authenticated routes, Hub base URL, lab registry, and frontend asset')
                print('No inference call and no existing Jupyter process was changed.')
            finally:
                # Let Jupyter close its own child operations before Windows removes
                # their working directory. Never stop an existing user server.
                try:
                    shutdown = Request(f'http://127.0.0.1:{port}{prefix}api/shutdown',
                        data=b'', headers={'Authorization': 'token '+token}, method='POST')
                    with open_request(shutdown, timeout=5):
                        pass
                except OSError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    else:
                        process.kill()
                    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    verify(parser.parse_args().wheel)


if __name__ == '__main__':
    main()

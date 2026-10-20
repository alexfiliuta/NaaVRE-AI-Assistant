"""Compile the frontend and normalize generated asset paths for Linux wheels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def normalize_bundle(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding='utf-8'))
    build = metadata['jupyterlab']['_build']
    build['load'] = build['load'].replace('\\', '/')
    path.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'jupyter', 'labextension', 'build', '.']
    if args.development:
        command += ['--development', 'True']
    subprocess.run(command, cwd=project, check=True)
    normalize_bundle(project/'naavre_ai_assistant/labextension/package.json')


if __name__ == '__main__':
    main()

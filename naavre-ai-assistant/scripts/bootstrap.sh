#!/usr/bin/env bash
set -euo pipefail
project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(CDPATH= cd -- "$project_root/.." && pwd)"
python_command="python3"
skip_frontend=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                printf '%s\n' '--python requires an interpreter command or path.' >&2
                exit 2
            fi
            python_command="$2"; shift 2 ;;
        --skip-frontend) skip_frontend=true; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ ! -f "$project_root/.venv/bin/python" ]]; then
    "$python_command" -m venv "$project_root/.venv"
fi
venv_python="$project_root/.venv/bin/python"
"$venv_python" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
"$venv_python" -m pip install --upgrade pip setuptools wheel
"$venv_python" -m pip install -e "${project_root}[test,build,notebook,analysis]"
if [[ "$skip_frontend" == false ]]; then
    export PATH="$project_root/.venv/bin:$PATH"
    (cd "$project_root" && npm ci && npm run build:all)
elif [[ ! -f "$project_root/naavre_ai_assistant/labextension/package.json" ]]; then
    printf '%s\n' '--skip-frontend requires an existing prebuilt frontend bundle.' >&2
    exit 1
fi
# A normal installation is required for Jupyter data_files; editable metadata alone
# does not install the frontend assets or the server auto-enable configuration.
"$venv_python" -m pip install --no-deps --no-build-isolation --force-reinstall "$project_root"
"$venv_python" -m pip check
printf '%s\n' 'Environment ready. Run naavre-ai-assistant/.venv/bin/python naavre-ai-assistant/scripts/start_demo.py.'

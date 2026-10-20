"""Launch the local demo using this workspace's environment, from any directory.

For hosted HTTPS deployments, configure and start Jupyter directly. This local
launcher intentionally selects loopback inference and never creates an SSH tunnel.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess


APPLICATION = Path(__file__).resolve().parents[1]
WORKSPACE = APPLICATION.parent


def demo_environment(model: str, labs: str, think: str) -> dict[str, str]:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}(,[a-z][a-z0-9_-]{0,63})*", labs):
        raise ValueError("Labs must be comma-separated lab identifiers.")
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    env["VIRTUAL_ENV"] = str(APPLICATION / ".venv")
    for name in ("AGENT_MODEL", "ANALYSIS_MODEL", "CODE_MODEL", "REMOTE_HOST", "CA_FILE",
                 "CLIENT_CERT", "CLIENT_KEY", "API_KEY_FILE"):
        env.pop("LOCAL_LLM_" + name, None)
    for name in ("NAAVRE_ENV_FILE", "NAAVRE_AGENT_STATE_DIR"):
        env.pop(name, None)
    env.update(
        LOCAL_LLM_BASE_URL="http://127.0.0.1:11434/v1",
        LOCAL_LLM_MODEL=model,
        LOCAL_LLM_THINK=think,
        LOCAL_LLM_CONTEXT_WINDOW="16384",
        LOCAL_LLM_TIMEOUT_SECONDS="unlimited",
        LOCAL_LLM_MAX_TOKENS="unlimited",
        NAAVRE_LABS_DIR=str(WORKSPACE / "Labs" if (WORKSPACE / "Labs").is_dir()
                            else WORKSPACE / "Testing" / "fixtures" / "labs"),
        NAAVRE_ALLOWED_LABS=labs,
        NAAVRE_AGENT_TOOLS="list_files,inspect_dataset,search_knowledge,list_skills",
        NAAVRE_AGENT_MAX_STEPS="12",
        NAAVRE_AGENT_RUN_SECONDS="off",
        NAAVRE_AGENT_OPERATION_SECONDS="off",
    )
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3-coder:30b", "qwen3.8:27b"), default="qwen3.8:27b")
    parser.add_argument("--labs", default="plants")
    parser.add_argument("--think", choices=("auto", "true", "false"), default="false")
    parser.add_argument("--check", action="store_true", help="Check configuration without contacting the model")
    args = parser.parse_args(argv)
    python = APPLICATION / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        parser.error("Project environment missing. Run naavre-ai-assistant/scripts/bootstrap.ps1 or naavre-ai-assistant/scripts/bootstrap.sh first.")
    try:
        env = demo_environment(args.model, args.labs, args.think)
    except ValueError as error:
        parser.error(str(error))
    command = ([str(python), "-m", "naavre_ai_assistant", "check"] if args.check else
               [str(python), "-m", "jupyterlab", "--no-browser", "--ip=127.0.0.1", "--port=8888",
                "--ServerApp.port_retries=0", f"--ServerApp.root_dir={WORKSPACE}",
                "--LabApp.extension_manager=readonly"])
    return subprocess.run(command, cwd=WORKSPACE, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

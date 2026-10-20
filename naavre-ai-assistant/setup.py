from pathlib import Path
import sys

from setuptools import find_packages, setup


HERE = Path(__file__).parent.resolve()
LABEXTENSION = HERE / "naavre_ai_assistant" / "labextension"

if any(command in sys.argv for command in ("bdist_wheel", "sdist")) and not (LABEXTENSION / "package.json").is_file():
    raise RuntimeError(
        "The JupyterLab frontend bundle is missing. Run 'npm install' and "
        "'npm run build:all' before building a release artifact."
    )


def data_files_for(source: Path, destination: str) -> list[tuple[str, list[str]]]:
    if not source.exists():
        return []

    grouped: dict[str, list[str]] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix != ".map":
            relative_parent = path.parent.relative_to(source)
            target = str(Path(destination) / relative_parent)
            # setuptools requires data-file sources to be relative to setup.py.
            # Absolute Windows paths break editable installs and wheel builds.
            grouped.setdefault(target, []).append(path.relative_to(HERE).as_posix())
    return sorted(grouped.items())


data_files = data_files_for(
    LABEXTENSION,
    "share/jupyter/labextensions/naavre-ai-assistant",
)
data_files += data_files_for(
    HERE / "jupyter-config",
    "etc/jupyter",
)

setup(
    name="naavre-ai-assistant",
    version="0.2.2",
    description="JupyterLab research coworker with lab-scoped MCP tools and self-hosted Ollama",
    long_description="JupyterLab research assistant with a Python/Tornado Jupyter Server backend, lab-scoped MCP tools and self-hosted Ollama inference.",
    long_description_content_type="text/plain",
    author="Alexandru Filiuta",
    license="BSD-3-Clause",
    python_requires=">=3.10",
    packages=find_packages(include=["naavre_ai_assistant"]),
    package_data={"naavre_ai_assistant": ["labextension/package.json", "labextension/static/*.js",
                  "labextension/static/*.json", "labextension/static/*.txt", "labextension/static/*.css"]},
    include_package_data=False,
    data_files=data_files,
    install_requires=[
        "jupyter_server>=2.14,<3",
        "jupyterlab>=4.2,<5",
        "python-dotenv>=1.0,<2",
        "mcp==2.2.0",
        "jsonschema>=4.20,<5",
        "pypdf>=6.18,<7",
    ],
    extras_require={
        "test": ["pytest>=8,<9", "pytest-asyncio>=0.23,<1", "nbclient>=0.10,<1",
                 "nbformat>=5.10,<6", "cryptography>=42"],
        "build": ["build>=1.2,<2", "wheel>=0.43"],
        "notebook": ["ipykernel>=6.29,<8"],
        "analysis": ["numpy>=1.26,<3", "pandas>=2,<3", "matplotlib>=3.8,<4"],
    },
    entry_points={"console_scripts": ["naavre-ai-assistant=naavre_ai_assistant.__main__:main"]},
    classifiers=[
        "Framework :: Jupyter",
        "Framework :: Jupyter :: JupyterLab :: 4",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

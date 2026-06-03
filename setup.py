from pathlib import Path
import sys

from setuptools import find_packages, setup


HERE = Path(__file__).parent.resolve()
LABEXTENSION = HERE / "naavre_ai_assistant" / "labextension"

if any(command in sys.argv for command in ("bdist_wheel", "sdist")) and not LABEXTENSION.exists():
    raise RuntimeError(
        "The JupyterLab frontend bundle is missing. Run 'npm install' and "
        "'npm run build:all' before building a release artifact."
    )


def data_files_for(source: Path, destination: str) -> list[tuple[str, list[str]]]:
    if not source.exists():
        return []

    grouped: dict[str, list[str]] = {}
    for path in source.rglob("*"):
        if path.is_file():
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
    version="0.1.0",
    description="AI-assisted draft-cell generation and repair for NaaVRE workflows",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Alexandru Filiuta",
    license="BSD-3-Clause",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    data_files=data_files,
    install_requires=[
        "jupyter_server>=2.14,<3",
        "jupyterlab>=4.2,<5",
        "openai>=1.30,<2",
        "python-dotenv>=1.0,<2",
    ],
    extras_require={
        "test": ["pytest>=8,<9", "pytest-asyncio>=0.23,<1"],
    },
    classifiers=[
        "Framework :: Jupyter",
        "Framework :: Jupyter :: JupyterLab :: 4",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

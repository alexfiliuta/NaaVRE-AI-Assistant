"""Allow the test suite to run from any working directory in a source checkout."""
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
for directory in (WORKSPACE / "naavre-ai-assistant", WORKSPACE / "Testing"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
